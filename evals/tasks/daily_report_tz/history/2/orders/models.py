from dataclasses import dataclass


@dataclass
class Order:
    id: str
    created: int          # Unix 時間(秒)
    amount: int           # 金額(元)
    status: str = "paid"  # paid / cancelled / refunded
