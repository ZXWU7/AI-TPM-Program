# Zhiwen Zhu 2024/04/22    zhiwenzhu@shu.edu.cn
from Auto_scan_class_V0 import *


if __name__ == '__main__':
    
    nanonis = Mustard_AI_Nanonis()
    # get current scan prameters
    ScanFrame = nanonis.ScanFrameGet()
    ScanBuffer = nanonis.ScanBufferGet()
    
    nanonis.nanocoodinate = (ScanFrame['center_x'], ScanFrame['center_y'])
    nanonis.Scan_edge = ScanFrame['width']
    nanonis.scan_square_Buffer_pix = ScanBuffer['Pixels']
    
    nanonis.monitor_thread_activate()                                                # activate the monitor thread
    
    image_t = nanonis.batch_scan_producer(nanonis.nanocoodinate, nanonis.Scan_edge, nanonis.scan_square_Buffer_pix, 0)
    # start the scan loop
    while True:
        
        image_t1 = nanonis.batch_scan_producer(nanonis.nanocoodinate, nanonis.Scan_edge, nanonis.scan_square_Buffer_pix, 0)
    
    
    pass
    