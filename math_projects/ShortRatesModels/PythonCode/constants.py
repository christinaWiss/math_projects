import math
# in this class all constants are being listed

start_model = 0
end_model = 1
width = 5000
gather_force = (end_model-start_model) * width
gather_unit = [i/width for i in list(range(start_model, end_model*width))]
gather_unit_without_zero = [i/width for i in list(range(start_model+1, end_model*width))]
pointsize = 80/math.log(width)

