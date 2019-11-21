#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019-11-20 16:08
@Author  : liuqingchen
@Email   : liuqingchen@chinamobile.com
@File    : MyLog.py
"""
import sys,logging
from wsgilog import WsgiLog,LogStdout
import config

class Log(WsgiLog):
    def __init__(self, application):
        WsgiLog.__init__(
            self,
            application,
            logformat = config.logformat,
            datefmt = config.datefmt,
            tofile = True,
            toprint = True,
            # tostream = True,
            file = config.log_file,
            interval = config.log_interval,
            backups = config.log_backups
        )
        sys.stdout = LogStdout(self.logger, logging.INFO)
        sys.stderr = LogStdout(self.logger, logging.ERROR)
