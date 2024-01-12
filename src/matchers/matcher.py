from src.frame import Frame


class Matcher():
    matcher_type: str 
    
    def generate_cost_matrix(self, f1: Frame, f2: Frame, normalize: bool = False):
        raise NotImplementedError
