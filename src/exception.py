import sys


def get_error_message(error: Exception, error_detail: sys) -> str:
    """
    Extract detailed information from an exception.
    """

    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return str(error)

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        f"Error occurred in file [{file_name}] "
        f"at line [{line_number}]: {str(error)}"
    )

    return error_message


class BankChurnException(Exception):

    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)

        self.error_message = get_error_message(
            error_message,
            error_detail
        )

    def __str__(self):
        return self.error_message