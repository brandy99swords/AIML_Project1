import sys

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occured in python script [{0}] at line [{1}]: {2}".format(
        file_name, exc_tb.tb_lineno, str(error)   # <-- fixed tb_lineno
    )

    return error_message


class custom_exception(Exception):
    def __init__(self, error_message, error_detail):
        """
        Initialize the custom exception with:
        - error_message: the original error or a string description
        - error_detail: usually 'sys', used to extract the traceback info
        """
        # ✅ Correct use of super
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
