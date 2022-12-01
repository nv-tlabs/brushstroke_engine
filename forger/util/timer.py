# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import time


class QuickTimer(object):
    def __init__(self):
        self.timers = {}
        self.lastKey = None

    def start(self, key):
        if key not in self.timers:
            self.timers[key] = {'total': 0}
        self.timers[key]['start'] = time.time()
        self.lastKey = key

    def stop(self, key=None):
        if not key:
            key = self.lastKey

        if self.timers[key]['start'] is not None:
            self.timers[key]['total'] += time.time() - self.timers[key]['start']
            self.timers[key]['start'] = None

    def summary(self):
        summ = [(x[1]['total'], x[0]) for x in self.timers.items()]
        summ.sort()
        return '\n'.join(['TIMING %s %0.5f' % (x[1], x[0]) for x in summ])