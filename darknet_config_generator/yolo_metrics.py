"""
Loss Descriptors

@author: Abdullahi S. Adamu
"""

from darknet_config_generator.common import NL


class Loss:
    """
    Base Loss Class
    """
    def __init__(self, loss_type=''):
        self.__HEADER__ = '[cost]'
        self.loss_type = loss_type

    def export(self, file_obj):
        """ export loss """
        file_obj.write(f'{NL}')
        file_obj.write(f'{self.__HEADER__}{NL}')
        file_obj.write(f'type={self.loss_type}{NL}')
        file_obj.write(NL)