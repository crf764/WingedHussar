import math

def age_buckets2(x):
    if x < 15: 
        return '14 år og under'
    if x >=80:
        return '80 år og derover'
    
    if x >= 15 and x < 25:
        return f'{int(x)} år'
    elif x > 24 and x < 30:
        return '25-29 år'
    else:
        #runder ned til den nærmeste 10'er
        x_low = math.floor(x/10)*10
        return f'{x_low}-{x_low+9} år'

age_buckets2(60)