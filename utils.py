import math

def hooke(x1, y1, x2, y2, natural_length, k):    

    xdiff = x2 - x1
    ydiff = y2 - y1
    current_length = math.sqrt(xdiff**2 + ydiff**2)
    abs_force = k * (current_length - natural_length)
    fx1 = abs_force* xdiff / current_length
    fx2 = -fx1
    fy1 = abs_force* ydiff / current_length
    fy2 = -fy1

    # May be easier to return singular object if numpy allows multiple assignment/alteration
    return fx1, fy1, fx2, fy2
