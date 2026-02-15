import torch
import torch.nn as nn

def calc_inverse(T: torch.Tensor) -> torch.Tensor:
    B, H, C, _ = T.size()
    dtype = T.dtype
    cur = 1
    res = T + torch.eye(C, device=T.device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # (B, H, C, C)
    return torch.linalg.inv(res.float()).to(dtype)
    # res = -T + torch.eye(C, device=T.device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # (B, H, C, C)
    # mu = T.float() @ T.float() # (B, H, C, C)
    # res = res.float()  # (B, H, C, C)
    # while True:
    #     if cur >= C:
    #         break
    #     res = res + (res @ mu)
    #     mu = mu @ mu
    #     cur = cur * 2 + 1
    # return res.to(dtype)
    eye = torch.eye(C, device=T.device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    # (I + T)^{-1} = sum_{s=0}^{C-1} (-1)^s T^s = sum_{s=0}^{C-1} A^s, A = -T
    res = eye.float()  # (B, H, C, C)
    mu = -T.float()  # (B, H, C, C), A^{1}
    while cur < C:
        res = res + (res @ mu)
        mu = mu @ mu
        cur = cur * 2
    return res.to(dtype)


class CalcInverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, T: torch.Tensor) -> torch.Tensor:
        # Compute Y = (I + T)^{-1} using the existing series-based method
        Y = calc_inverse(T)
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (Y,) = ctx.saved_tensors
        # dY = -Y dT Y  => grad_T = -(Y^T @ grad_output @ Y^T)
        Y_t = Y.transpose(-2, -1)
        grad_T = -(Y_t @ grad_output @ Y_t)
        return grad_T


def calc_inverse_autograd(T: torch.Tensor) -> torch.Tensor:
    return CalcInverseFunction.apply(T)

def delta_rule(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    init_state: torch.Tensor
):
    """
    Implements the delta rule attention mechanism.

    Args:
        k: Key tensor of shape (B, L, H, D)
        q: Query tensor of shape (B, L, H, D)
        v: Value tensor of shape (B, L, H, D)
        beta: Gating tensor of shape (B, L, H)
        init_state: Initial state tensor of shape (B, H, D, D)
    
    Returns:
        Output tensor of shape (B, L, H, D)
    """
    k = k.transpose(1, 2)  # (B, H, L, D)
    q = q.transpose(1, 2)  # (B, H, L, D)
    v = v.transpose(1, 2)  # (B, H, L, D)
    beta = beta.transpose(1, 2)  # (B, H, L)

    B, H, L, D = k.size()
    CHUNK_SIZE = 64  # to save memory

    state = init_state  # (B, H, D, D)
    o = []
    for i in range(0, L, CHUNK_SIZE):
        i_end = min(i + CHUNK_SIZE, L)
        k_chunk = k[:, :, i:i_end, :]  # (B, H, C, D)
        q_chunk = q[:, :, i:i_end, :]  # (B, H, C, D)
        v_chunk = v[:, :, i:i_end, :]  # (B, H, C, D)
        beta_chunk = beta[:, :, i:i_end]  # (B, H, C)
        beta_k_chunk = beta_chunk.unsqueeze(-1) * k_chunk  # (B, H, C, D)

        T = (k_chunk @ beta_k_chunk.transpose(-2, -1)).tril(-1) # (B, H, C, C)
        W = k_chunk @ state  # (B, H, C, D)
        inv_T_I = calc_inverse(T)
        # inv_T_I = calc_inverse_autograd(T)

        # check https://kexue.fm/archives/11563
        assert inv_T_I.all() >= -1.0 and inv_T_I.all() <= 1.0, "Inverse matrix has invalid values"

        u_chunk = inv_T_I @ (v_chunk - W)  # (B, H, C, D)

        a_chunk = (q_chunk @ beta_k_chunk.transpose(-2, -1)).tril()  # (B, H, C, C)
        o_chunk = a_chunk @ u_chunk + q_chunk @ state # (B, H, C, D)
        
        state = state + beta_k_chunk.transpose(-2, -1) @ u_chunk

        o.append(o_chunk.view(B, H, i_end - i, D))
    
    o = torch.concat(o, dim=2)  # (B, H, L, D)
    o = o.transpose(1, 2)  # (B, L, H, D)
    return o

def brute_force_delta_rule(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    init_state: torch.Tensor
):
    B, L, H, D = k.size()
    k = k.transpose(1, 2)  # (B, H, L, D)
    q = q.transpose(1, 2)  # (B, H, L, D)
    v = v.transpose(1, 2)  # (B, H, L, D)
    beta = beta.transpose(1, 2)  # (B, H, L)

    o = torch.zeros(B, H, L, D, device=k.device, dtype=k.dtype)
    state = init_state
    for t in range(L):
        k_t = k[:, :, t, :].unsqueeze(-2)  # (B, H, 1, D)
        q_t = q[:, :, t, :].unsqueeze(-2)  # (B, H, 1, D)
        v_t = v[:, :, t, :].unsqueeze(-2)  # (B, H, 1, D)
        beta_t = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

        state = state - beta_t * k_t.transpose(-2, -1) @ (k_t @ state)  # (B, H, D, D)
        state = state + beta_t * (k_t.transpose(-2, -1) @ v_t)  # (B, H, D, D)

        o_t = (q_t @ state).reshape(B, H, 1, D)  # (B, H, 1, D)
        o[:, :, t, :] = o_t.squeeze(-2)

    o = o.transpose(1, 2)  # (B, L, H, D)
    return o

def test_delta_rule():
    B, L, H, D = 2, 2048, 4, 8
    k = torch.randn(B, L, H, D)
    k = k / torch.norm(k, p=2, dim=-1, keepdim=True)
    # k = torch.ones(B, L, H, D)
    q = torch.randn(B, L, H, D)
    q = q / torch.norm(q, p=2, dim=-1, keepdim=True)
    # q = torch.ones(B, L, H, D)
    v = torch.randn(B, L, H, D)
    beta = torch.sigmoid(torch.randn(B, L, H))
    # beta = torch.ones_like(torch.randn(B, L, H))
    init_state = torch.zeros(B, H, D, D)

    # print(f"k: {k}")
    # print(f"q: {q}")
    # print(f"v: {v}")
    # print(f"beta: {beta}")

    o = delta_rule(k, q, v, beta, init_state)
    o_brute = brute_force_delta_rule(k, q, v, beta, init_state)
    # print(o)
    # print(o_brute)
    assert torch.allclose(o, o_brute, atol=1e-4), "Delta rule implementation does not match brute force!"
    print("Output shape:", o.shape)  # Should be (B, L, H, D)

if __name__ == "__main__":
    test_delta_rule()