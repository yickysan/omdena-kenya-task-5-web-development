import sys
from typing import Protocol
from OmdenaKenyaRoadAccidents.logger import logging


def error_msg_details(error: str, error_details:sys)-> str:
    """
    Function to Format Custom Exception Message
    """
    _,_, exc_tb = error_details.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    
    error_message = f"Error occured in python script name: {filename} \n line number: [{line_no}] \n\
error message: {str(error)}"
    return error_message


class DataIngestionError(Exception):
    def __init__(self, error_message: str, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_msg_details(error=error_message, error_details=error_details)

    def __str__(self) -> str:
        return self.error_message
    
    

class DataTransformationError(Exception):
    def __init__(self, error_message: str, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_msg_details(error=error_message, error_details=error_details)

    def __str__(self) -> str:
        return self.error_message


class ModelTrainerError(Exception):
    def __init__(self, error_message:str, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_msg_details(error=error_message, error_details=error_details)

    def __str__(self) -> str:
        return self.error_message