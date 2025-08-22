import gymnasium as gym


"""
action space 9 discrete :
    - -1 for nothing
    - 0 rad for angle 0 or 360° or 2PI for angle 360
    - 45° or PI/4 for angle 45
    - 90° or PI/2 for angle 90
    - 135° or 3PI/4 for angle 135
    - 180° or PI for angle 180
    - 225° or 5PI/4 for angle 225
    - 270° or 3PI/2 for angle 270
    - 315° or 5PI/3 for angle 315
    
observation space : 

rewards:
    - -1 per turn without checkpoint
    - +20 when end
    - +5 when checkpoint

end :
    - lap over : i.e all checkpoint validated in right order 3 times
    - 100 round without reaching a checkpoint
"""
