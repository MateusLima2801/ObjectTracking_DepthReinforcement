from __future__ import annotations
import math

class Vector():
    def __init__(self, x:float, y:float):
        self.x = x
        self.y = y

    def is_null(self) -> bool:
        return self.x == 0 and self.y == 0
    
    def dot_product(self, other: Vector) -> float:
        return self.x * other.x + self.y * other.y
    
    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)
    
    def cosin(self, other: Vector) -> float:
        if self.is_null() or other.is_null():
            raise ArithmeticError("0 normed vector.")
        else: return self.dot_product(other) / (self.norm() * other.norm())
