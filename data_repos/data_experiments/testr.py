# import importlib
# rpy2 = importlib.util.find_spec("rpy2")
from rpy2 import robjects
pi = robjects.r['pi']
pi