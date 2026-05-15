import torch
import torch.optim as optim


class BasisRotation(optim.Optimizer):
    """Adam with Basis Rotation (Algorithm 1 in the paper).

    Transforms the optimization space to align with the Hessian eigenbasis,
    mitigating the adverse effects of gradient staleness in asynchronous
    pipeline-parallel training.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        betas: Coefficients for computing running averages of gradient and its square.
        shampoo_beta: EMA coefficient for covariance accumulation. If < 0, uses betas[1].
        eps: Term added to denominator for numerical stability.
        weight_decay: Weight decay (L2 penalty).
        precondition_frequency: How often to update the rotation basis (T_freq in paper).
        rotation_geometry: Rotation geometry — "bi" (bilateral/two-sided) or
                           "uni" (unilateral/one-sided). Corresponds to G in Algorithm 2.
        approx_source: Approximation source for eigenbasis estimation — "2nd" (second-order
                       covariance, S=2^nd) or "1st" (first-order gradient, S=1^st).
                       Corresponds to S in Algorithm 2.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        rotation_geometry: str = "bi",   # "uni" (unilateral) | "bi" (bilateral)
        approx_source: str = "2nd",      # "1st" (gradient) | "2nd" (covariance)
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "rotation_geometry": rotation_geometry,
            "approx_source": approx_source,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.dim() != 2:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    self.init_preconditioner_state(state, grad, group)

                state["step"] += 1

                # --- 1. Update Preconditioner (Eigenbasis-Estimation, Algorithm 2) ---
                if group["approx_source"] == "2nd":
                    self.accumulate_covariance(state, grad, group)

                should_update_basis = (
                    state["step"] % group["precondition_frequency"] == 0
                    or state["Q_L"] is None
                )

                if should_update_basis:
                    # Target momentum for basis estimation (lookahead by one Adam step)
                    beta1 = group["betas"][0]
                    curr_momentum = state["exp_avg"]
                    target_momentum = beta1 * curr_momentum + (1 - beta1) * grad
                    self.update_preconditioner(state, grad, target_momentum, group)

                # --- 2. Compute Update in Rotated Space ---
                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Rotate gradient and momentum into the eigenbasis
                grad_rotated = self.project(grad, state)
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                rotated_momentum = self.project(exp_avg, state)

                rotated_sq = grad_rotated.square()
                state["exp_avg_sq"].mul_(beta2).add_(rotated_sq, alpha=(1.0 - beta2))
                denom = state["exp_avg_sq"].sqrt().add_(group["eps"])

                # --- 3. Apply Update and Project Back ---
                step_size = group["lr"]
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                update_rotated = rotated_momentum / denom
                norm_grad = self.project_back(update_rotated, state)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                p.add_(norm_grad, alpha=-step_size)

        return loss

    def project(self, tensor, state):
        """Projects tensor into the eigenbasis (U^T G V)."""
        out = tensor
        if state["Q_L"] is not None:
            out = state["Q_L"].T @ out
        if state["Q_R"] is not None:
            out = out @ state["Q_R"]
        return out

    def project_back(self, tensor, state):
        """Projects tensor back to the original parameter space (U G V^T)."""
        out = tensor
        if state["Q_R"] is not None:
            out = out @ state["Q_R"].T
        if state["Q_L"] is not None:
            out = state["Q_L"] @ out
        return out

    def init_preconditioner_state(self, state, grad, group):
        rows, cols = grad.shape
        geom = group["rotation_geometry"]

        state["Q_L"] = None
        state["Q_R"] = None
        state["GG_L"] = None
        state["GG_R"] = None
        state["dims"] = []

        if group["approx_source"] == "2nd":
            use_left  = (geom == "bi") or (geom == "uni" and rows < cols)
            use_right = (geom == "bi") or (geom == "uni" and rows >= cols)
            if use_left:
                state["GG_L"] = torch.zeros(rows, rows, device=grad.device)
            if use_right:
                state["GG_R"] = torch.zeros(cols, cols, device=grad.device)

    def accumulate_covariance(self, state, grad, group):
        """EMA update of gradient covariance matrices (S=2nd)."""
        beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else group["betas"][1]
        if state["GG_L"] is not None:
            state["GG_L"].lerp_(grad @ grad.T, 1 - beta)
        if state["GG_R"] is not None:
            state["GG_R"].lerp_(grad.T @ grad, 1 - beta)

    def update_preconditioner(self, state, grad, momentum, group):
        """Compute new rotation matrices U, V via Eigenbasis-Estimation (Algorithm 2)."""
        rows, cols = grad.shape
        geom = group["rotation_geometry"]
        src = group["approx_source"]

        use_left  = (geom == "bi") or (geom == "uni" and rows < cols)
        use_right = (geom == "bi") or (geom == "uni" and rows >= cols)

        is_warm_start = (state["Q_L"] is not None) if use_left else True
        if use_right:
            is_warm_start = is_warm_start and (state["Q_R"] is not None)

        mat_list   = []
        orth_list  = []
        target_dims = []

        if src == "2nd":
            if use_left:
                mat_list.append(state["GG_L"])
                orth_list.append(
                    state["Q_L"] if state["Q_L"] is not None
                    else torch.eye(rows, device=grad.device)
                )
                target_dims.append(0)
            if use_right:
                mat_list.append(state["GG_R"])
                orth_list.append(
                    state["Q_R"] if state["Q_R"] is not None
                    else torch.eye(cols, device=grad.device)
                )
                target_dims.append(1)

        elif src == "1st":
            target_matrix = momentum.float()
            if not is_warm_start:
                mat_list = [target_matrix]
            else:
                if use_left:
                    qr_curr = state["Q_R"] if state["Q_R"] is not None else torch.eye(cols, device=grad.device)
                    mat_list.append(target_matrix)
                    orth_list.append(qr_curr)
                    target_dims.append(0)
                if use_right:
                    ql_curr = state["Q_L"] if state["Q_L"] is not None else torch.eye(rows, device=grad.device)
                    mat_list.append(target_matrix.T)
                    orth_list.append(ql_curr)
                    target_dims.append(1)

        state["GG"]   = mat_list
        state["Q"]    = orth_list
        state["dims"] = target_dims

        if is_warm_start:
            new_bases = self.get_orthogonal_matrix_QR(state, src)
        else:
            new_bases = self.get_orthogonal_matrix(state["GG"], src)

        idx = 0
        if src == "2nd":
            if use_left:
                state["Q_L"] = new_bases[idx].to(grad.dtype)
                idx += 1
            if use_right:
                state["Q_R"] = new_bases[idx].to(grad.dtype)
                idx += 1

        elif src == "1st":
            if not is_warm_start:
                U_mat = new_bases[0].to(grad.dtype)
                V_mat = new_bases[1].to(grad.dtype) if len(new_bases) > 1 else None
                if use_left:
                    state["Q_L"] = U_mat
                if use_right and V_mat is not None:
                    state["Q_R"] = V_mat
            else:
                if use_left:
                    state["Q_L"] = new_bases[idx].to(grad.dtype)
                    idx += 1
                if use_right:
                    state["Q_R"] = new_bases[idx].to(grad.dtype)
                    idx += 1

        del state["GG"]
        del state["Q"]
        del state["dims"]

    def get_orthogonal_matrix(self, mat, approx_source):
        """Full SVD/eigendecomposition for cold-start basis initialization."""
        matrix = []
        for m in mat:
            if len(m) == 0:
                matrix.append([])
                continue
            matrix.append(m.float())

        final = []
        for m in matrix:
            if len(m) == 0:
                final.append([])
                continue

            if approx_source == "1st":
                try:
                    U, _, Vh = torch.linalg.svd(m, full_matrices=True)
                except Exception:
                    U, _, Vh = torch.linalg.svd(m.to(torch.float64), full_matrices=True)
                    U  = U.to(m.dtype)
                    Vh = Vh.to(m.dtype)
                final.append(U)
                final.append(Vh.T)
            else:  # "2nd"
                try:
                    _, Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))
                except Exception:
                    _, Q = torch.linalg.eigh(
                        m.to(torch.float64) + 1e-30 * torch.eye(m.shape[0], device=m.device)
                    )
                    Q = Q.to(m.dtype)
                final.append(torch.flip(Q, [1]))
        return final

    def get_orthogonal_matrix_QR(self, state, approx_source):
        """Warm-started QR-based power iteration for efficient basis updates."""
        precond_list = state["GG"]
        orth_list    = state["Q"]
        target_dims  = state["dims"]

        matrix      = []
        orth_matrix = []
        for m, o in zip(precond_list, orth_list):
            if len(m) == 0:
                matrix.append([])
                orth_matrix.append([])
                continue
            matrix.append(m.float())
            orth_matrix.append(o.float())

        exp_avg_sq = state["exp_avg_sq"]

        final = []
        for m, o, dim_idx in zip(matrix, orth_matrix, target_dims):
            if len(m) == 0:
                final.append([])
                continue

            if approx_source == "1st":
                if dim_idx == 1:
                    # Update Q_R: G^T U = V Sigma  (power iteration on G^T U)
                    u_temp = m @ o
                    est_eig  = torch.sum(u_temp ** 2, dim=0)
                    sort_idx = torch.argsort(est_eig, descending=True)
                    k = sort_idx.shape[0]
                    # Permute second-moment state to stay aligned with new basis ordering
                    active = exp_avg_sq[:k, :]
                    rest   = exp_avg_sq[k:, :]
                    active = active.index_select(0, sort_idx)
                    exp_avg_sq = torch.cat([active, rest], dim=0)
                    o = o[:, sort_idx]
                    u_temp = m @ o
                    Q, _ = torch.linalg.qr(u_temp, mode="complete")

                else:  # dim_idx == 0
                    # Update Q_L: G V = U Sigma  (power iteration on G V)
                    u_unnorm = m @ o
                    est_eig  = torch.sum(u_unnorm ** 2, dim=0)
                    sort_idx = torch.argsort(est_eig, descending=True)
                    k = sort_idx.shape[0]
                    # Permute second-moment state columns
                    active = exp_avg_sq[:, :k]
                    rest   = exp_avg_sq[:, k:]
                    active = active.index_select(1, sort_idx)
                    exp_avg_sq = torch.cat([active, rest], dim=1)
                    o = o[:, sort_idx]
                    u_unnorm = m @ o
                    Q, _ = torch.linalg.qr(u_unnorm, mode="complete")

            else:  # "2nd"
                est_eig  = torch.diag(o.T @ m @ o)
                sort_idx = torch.argsort(est_eig, descending=True)
                exp_avg_sq = exp_avg_sq.index_select(dim_idx, sort_idx)
                o = o[:, sort_idx]
                power_iter = m @ o
                Q, _ = torch.linalg.qr(power_iter)

            final.append(Q)

        state["exp_avg_sq"] = exp_avg_sq

        return final
