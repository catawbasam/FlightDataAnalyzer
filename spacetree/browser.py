# -*- coding: utf-8 -*-
################################################################################

'''
Backport of additional web browser support.
'''

################################################################################
# Imports


import os
import sys

from webbrowser import BackgroundBrowser, UnixBrowser, register, _iscommand


################################################################################
# Exports


__all__ = ['register_additional_browsers']


################################################################################
# Classes


class Chrome(UnixBrowser):
    '''
    Launcher class for Google Chrome browser.
    '''
    remote_args = ['%action', '%s']
    remote_action = ''
    remote_action_newwin = '--new-window'
    remote_action_newtab = ''
    background = True


Chromium = Chrome


################################################################################
# Functions


def _register_xdg_open():
    '''
    '''
    if _iscommand('xdg-open'):
        register('xdg-open', None, BackgroundBrowser('xdg-open'))


def _register_gvfs_open():
    '''
    '''
    if 'GNOME_DESKTOP_SESSION_ID' in os.environ and _iscommand('gvfs-open'):
        register('gvfs-open', None, BackgroundBrowser('gvfs-open'))


def _register_google_chrome():
    '''
    '''
    for browser in ('google-chrome', 'chrome', 'chromium', 'chromium-browser'):
        if _iscommand(browser):
            register(browser, None, Chrome(browser))


def register_additional_browsers():
    '''
    '''
    if os.environ.get('DISPLAY') and sys.version_info[0:3] < (3, 3):
        _register_xdg_open()
        _register_gvfs_open()
        _register_google_chrome()


################################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
