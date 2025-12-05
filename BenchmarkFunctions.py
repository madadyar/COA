import numpy as np
import math

class BenchmarkFunctions:
    """
    Benchmark Functions Collection  
    Developed in Python 3.8+   
    Author and programmer: Mojtaba Madadyar
    e-Mail: mojtaba.madadyar@gmail.com
    """
    
    @staticmethod
    def get_function_details(F):
        """
        Get function details including lower bound, upper bound, dimension and objective function
        
        Parameters:
        F (str): Function name (F1 to F24)
        
        Returns:
        tuple: (lb, ub, dim, fobj)
        """
        switch = {
            'F1': {  # Sphere Function
                'lb': -100,
                'ub': 100,
                'dim': 30,
                'fobj': BenchmarkFunctions.F1
            },
            'F2': {  # Schwefel 2.22 Function
                'lb': -10,
                'ub': 10,
                'dim': 30,
                'fobj': BenchmarkFunctions.F2
            },
            'F3': {  # Schwefel 1.2 Function
                'lb': -100,
                'ub': 100,
                'dim': 30,
                'fobj': BenchmarkFunctions.F3
            },
            'F4': {  # Schwefel 2.21 Function
                'lb': -100,
                'ub': 100,
                'dim': 30,
                'fobj': BenchmarkFunctions.F4
            },
            'F5': {  # Brown Function
                'lb': -1,
                'ub': 4,
                'dim': 30,
                'fobj': BenchmarkFunctions.F5
            },
            'F6': {  # Alpine Function
                'lb': 0,
                'ub': 10,
                'dim': 30,
                'fobj': BenchmarkFunctions.F6
            },
            'F7': {  # Quartic Function
                'lb': -1.28,
                'ub': 1.28,
                'dim': 30,
                'fobj': BenchmarkFunctions.F7
            },
            'F8': {  # Schwefel Function
                'lb': -500,
                'ub': 500,
                'dim': 30,
                'fobj': BenchmarkFunctions.F8
            },
            'F9': {  # Rastrigin Function
                'lb': -5.12,
                'ub': 5.12,
                'dim': 30,
                'fobj': BenchmarkFunctions.F9
            },
            'F10': {  # Ackley Function
                'lb': -32,
                'ub': 32,
                'dim': 30,
                'fobj': BenchmarkFunctions.F10
            },
            'F11': {  # GRIEWANK Function
                'lb': -600,
                'ub': 600,
                'dim': 30,
                'fobj': BenchmarkFunctions.F11
            },
            'F12': {  # Salomon Function
                'lb': -100,
                'ub': 100,
                'dim': 30,
                'fobj': BenchmarkFunctions.F12
            },
            'F13': {  # Xin-She Yang Function
                'lb': -5,
                'ub': 5,
                'dim': 30,
                'fobj': BenchmarkFunctions.F13
            },
            'F14': {  # Ackley N. 2 Function
                'lb': -32,
                'ub': 32,
                'dim': 2,
                'fobj': BenchmarkFunctions.F14
            },
            'F15': {  # Kowalik Function
                'lb': -5,
                'ub': 5,
                'dim': 4,
                'fobj': BenchmarkFunctions.F15
            },
            'F16': {  # SIX-HUMP CAMEL Function
                'lb': -5,
                'ub': 5,
                'dim': 2,
                'fobj': BenchmarkFunctions.F16
            },
            'F17': {  # Branin's RCOS No.01 Function
                'lb': np.array([-5, 0]),
                'ub': np.array([10, 15]),
                'dim': 2,
                'fobj': BenchmarkFunctions.F17
            },
            'F18': {  # Branin's RCOS No.02 Function
                'lb': -2,
                'ub': 2,
                'dim': 2,
                'fobj': BenchmarkFunctions.F18
            },
            'F19': {  # Hartman 3 Function
                'lb': 0,
                'ub': 1,
                'dim': 3,
                'fobj': BenchmarkFunctions.F19
            },
            'F20': {  # Hartman 6 Function
                'lb': 0,
                'ub': 1,
                'dim': 6,
                'fobj': BenchmarkFunctions.F20
            },
            'F21': {  # Shekel 5 Function
                'lb': 0,
                'ub': 10,
                'dim': 4,
                'fobj': BenchmarkFunctions.F21
            },
            'F22': {  # Shekel 7 Function
                'lb': 0,
                'ub': 10,
                'dim': 4,
                'fobj': BenchmarkFunctions.F22
            },
            'F23': {  # Shekel 10 Function
                'lb': 0,
                'ub': 10,
                'dim': 4,
                'fobj': BenchmarkFunctions.F23
            },
            'F24': {  # Zakharov Function
                'lb': -5,
                'ub': 10,
                'dim': 30,
                'fobj': BenchmarkFunctions.F24
            }
        }
        
        if F in switch:
            details = switch[F]
            return details['lb'], details['ub'], details['dim'], details['fobj']
        else:
            raise ValueError(f"Function {F} not found!")
    
    # F1: Sphere Function
    @staticmethod
    def F1(x):
        return np.sum(x**2)
    
    # F2: Schwefel 2.22 Function
    @staticmethod
    def F2(x):
        return np.sum(np.abs(x)) + np.prod(np.abs(x))
    
    # F3: Schwefel 1.2 Function
    @staticmethod
    def F3(x):
        dim = len(x)
        o = 0
        for i in range(dim):
            o += np.sum(x[:i+1])**2
        return o
    
    # F4: Schwefel 2.21 Function
    @staticmethod
    def F4(x):
        return np.max(np.abs(x))
    
    # F5: Brown Function
    @staticmethod
    def F5(x):
        n = len(x)
        o = 0
        x_sq = x**2
        for i in range(n-1):
            o += x_sq[i]**(x_sq[i+1] + 1) + x_sq[i+1]**(x_sq[i] + 1)
        return o
    
    # F6: Alpine Function
    @staticmethod
    def F6(x):
        return np.sum(np.abs(x * np.sin(x) + 0.1 * x))
    
    # F7: Quartic Function
    @staticmethod
    def F7(x):
        dim = len(x)
        indices = np.arange(1, dim+1)
        return np.sum(indices * (x**4)) + np.random.rand()
    
    # F8: Schwefel Function
    @staticmethod
    def F8(x):
        return np.sum(-x * np.sin(np.sqrt(np.abs(x))))
    
    # F9: Rastrigin Function
    @staticmethod
    def F9(x):
        dim = len(x)
        return np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim
    
    # F10: Ackley Function
    @staticmethod
    def F10(x):
        dim = len(x)
        return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - \
               np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1)
    
    # F11: GRIEWANK Function
    @staticmethod
    def F11(x):
        dim = len(x)
        indices = np.arange(1, dim+1)
        return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(indices))) + 1
    
    # F12: Salomon Function
    @staticmethod
    def F12(x):
        x2 = x**2
        sumx2 = np.sum(x2)
        sqrtsx2 = np.sqrt(sumx2)
        return 1 - np.cos(2 * np.pi * sqrtsx2) + (0.1 * sqrtsx2)
    
    # F13: Xin-She Yang Function
    @staticmethod
    def F13(x):
        n = len(x)
        o = 0
        for i in range(n):
            o += np.random.rand() * (np.abs(x[i]) ** (i+1))
        return o
    
    # F14: Ackley N. 2 Function
    @staticmethod
    def F14(x):
        if len(x) != 2:
            raise ValueError('Ackley N. 2 function is only defined on a 2D space.')
        X, Y = x[0], x[1]
        return -200 * np.exp(-0.02 * np.sqrt(X**2 + Y**2))
    
    # F15: Kowalik Function
    @staticmethod
    def F15(x):
        aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
        bK = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
        return np.sum((aK - ((x[0] * (bK**2 + x[1] * bK)) / (bK**2 + x[2] * bK + x[3])))**2)
    
    # F16: SIX-HUMP CAMEL Function
    @staticmethod
    def F16(x):
        return 4 * (x[0]**2) - 2.1 * (x[0]**4) + (x[0]**6) / 3 + x[0] * x[1] - 4 * (x[1]**2) + 4 * (x[1]**4)
    
    # F17: Branin's RCOS No.01 Function
    @staticmethod
    def F17(x):
        return (x[1] - (x[0]**2) * 5.1 / (4 * (np.pi**2)) + 5 / np.pi * x[0] - 6)**2 + \
               10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10
    
    # F18: Branin's RCOS No.02 Function
    @staticmethod
    def F18(x):
        term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*(x[0]**2) - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
        term2 = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*(x[0]**2) + 48*x[1] - 36*x[0]*x[1] + 27*(x[1]**2))
        return term1 * term2
    
    # F19: Hartman 3 Function
    @staticmethod
    def F19(x):
        aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
        cH = np.array([1, 1.2, 3, 3.2])
        pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], 
                      [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
        o = 0
        for i in range(4):
            o -= cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :])**2)))
        return o
    
    # F20: Hartman 6 Function
    @staticmethod
    def F20(x):
        aH = np.array([[10, 3, 17, 3.5, 1.7, 8], 
                      [0.05, 10, 17, 0.1, 8, 14], 
                      [3, 3.5, 1.7, 10, 17, 8], 
                      [17, 8, 0.05, 10, 0.1, 14]])
        cH = np.array([1, 1.2, 3, 3.2])
        pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                      [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                      [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                      [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
        o = 0
        for i in range(4):
            o -= cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :])**2)))
        return o
    
    # F21: Shekel 5 Function
    @staticmethod
    def F21(x):
        aSH = np.array([[4, 4, 4, 4],
                       [1, 1, 1, 1],
                       [8, 8, 8, 8],
                       [6, 6, 6, 6],
                       [3, 7, 3, 7],
                       [2, 9, 2, 9],
                       [5, 5, 3, 3],
                       [8, 1, 8, 1],
                       [6, 2, 6, 2],
                       [7, 3.6, 7, 3.6]])
        cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        
        o = 0
        for i in range(5):
            o -= 1 / (np.sum((x - aSH[i, :])**2) + cSH[i])
        return o
    
    # F22: Shekel 7 Function
    @staticmethod
    def F22(x):
        aSH = np.array([[4, 4, 4, 4],
                       [1, 1, 1, 1],
                       [8, 8, 8, 8],
                       [6, 6, 6, 6],
                       [3, 7, 3, 7],
                       [2, 9, 2, 9],
                       [5, 5, 3, 3],
                       [8, 1, 8, 1],
                       [6, 2, 6, 2],
                       [7, 3.6, 7, 3.6]])
        cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        
        o = 0
        for i in range(7):
            o -= 1 / (np.sum((x - aSH[i, :])**2) + cSH[i])
        return o
    
    # F23: Shekel 10 Function
    @staticmethod
    def F23(x):
        aSH = np.array([[4, 4, 4, 4],
                       [1, 1, 1, 1],
                       [8, 8, 8, 8],
                       [6, 6, 6, 6],
                       [3, 7, 3, 7],
                       [2, 9, 2, 9],
                       [5, 5, 3, 3],
                       [8, 1, 8, 1],
                       [6, 2, 6, 2],
                       [7, 3.6, 7, 3.6]])
        cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
        
        o = 0
        for i in range(10):
            o -= 1 / (np.sum((x - aSH[i, :])**2) + cSH[i])
        return o
    
    # F24: Zakharov Function
    @staticmethod
    def F24(x):
        d = len(x)
        sum1 = 0
        sum2 = 0
        
        for ii in range(d):
            xi = x[ii]
            sum1 += xi**2
            sum2 += 0.5 * (ii + 1) * xi  # Note: ii+1 because Python indexing starts at 0
        
        return sum1 + sum2**2 + sum2**4
    
    # Utility function (not used in current implementation but included for completeness)
    @staticmethod
    def Ufun(x, a, k, m):
        return k * ((x - a)**m) * (x > a) + k * ((-x - a)**m) * (x < (-a))


# Example usage
if __name__ == "__main__":
    # Test the new functions
    for func_name in ['F21', 'F22', 'F23', 'F24']:
        lb, ub, dim, fobj = BenchmarkFunctions.get_function_details(func_name)
        print(f"Function {func_name}: lb={lb}, ub={ub}, dim={dim}")
        
        # Test the function with a random point
        x = np.random.uniform(lb, ub, dim)
        result = fobj(x)
        print(f"{func_name}({x[:min(5, dim)]}...) = {result}")
        print("-" * 50)