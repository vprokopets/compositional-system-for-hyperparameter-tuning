__doc__ = """
    Describes logic of selection algorithm based on Sobol sequences in Sobol space."""

import sobol_seq
import cProfile
import pstats
import io
pr = cProfile.Profile()


dimensionality = range(1, 10)
configuration_number = range(0, 10000)
res = {}
pr.enable()
for dim in dimensionality:
    seed = 0
    res_arr = []
    for _ in configuration_number:
        vector, seed = sobol_seq.i4_sobol(dim, seed)
        res_arr.append(vector)
    res.update({dim: res_arr})
        #print(vector)
pr.disable()
result = io.StringIO()
pstats.Stats(pr,stream=result).print_stats()
result=result.getvalue()

print(res.keys())
print(result)