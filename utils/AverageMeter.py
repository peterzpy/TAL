class AverageMeter(object):
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.num = 0

    def update(self, val, n = 1):
        self.val += val * n
        self.num += n
        self.avg = self.val / self.num