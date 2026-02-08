from dataclasses import dataclass

@dataclass
class CRFConfig:
    # CRF hyperparams (lbfgs thường tốt)
    algorithm: str = "lbfgs"
    c1: float = 0.1          # L1 regularization
    c2: float = 0.1          # L2 regularization
    max_iterations: int = 200
    all_possible_transitions: bool = True

    # feature window
    window: int = 2          # nhìn trái/phải 2 token

