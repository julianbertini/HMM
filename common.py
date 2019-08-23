import numpy as np


BACKWARD = "backward"
DEFAULT = "default"
STATE = "state"
OBS = "observation"
TIME = "time"


class Common():

    def __init__(self, N, T, A, B, O):
        self.N = N
        self.T = T + 2  # +2 for START and END states
        self.A_init = A
        self.A = A
        self.B = B
        self.O = O
        return None

    def update_parameters(self, A, B, O):
        self.A = A
        self.B = B
        self.O = O

    def initialize_dynamic_table(self, init_type="default"):

        table = np.zeros((self.T, self.N))

        if init_type == BACKWARD:
            table[self.T-1][self.N-1] = 1
            for i in range(1, self.N):
                table[self.T-2][i] = self.A[i][self.A.shape[1]-1]

        elif init_type == DEFAULT:
            table[0][0] = 1
            for j in range(1, self.N):
                table[1][j] = self.A[0][j] * self.B[j][self.O[0] - 1]

        else:
            raise TypeError("Please provide a valid init type:"
                            "e.g., backward.")

        return table

    def pretty_print(self, table, x_axis="state", y_axis="time", table_name="Table"):

        table_str = ''
        header_str = ''
        sep_str = ''
        table_str += '\n'

        header_str += '{:^11s} |'.format(table_name)
        sep_str += '{:^11s} |'.format('')
        for i in range(table.shape[1]):
            if x_axis == STATE:
                header_str += '{:1s} {:^9s} {:1s}'.format(
                    '|', 's = ' + str(i), '')
            elif x_axis == OBS:
                try:
                    header_str += '{:1s} {:^9s} {:1s}'.format(
                        '|', 'o = ' + str(i+1), '')
                except(IndexError):
                    raise IndexError(
                        "Table provided does not match observation array dimensions.")
            else:
                raise TypeError('Please enter a valid x-axis type.')

            sep_str += '{:1s} {:^9s} {:1s}'.format('|', '=+=', '')
        header_str += '||\n'
        sep_str += '||\n'

        table_str += header_str + sep_str

        for i in range(table.shape[0]):

            if y_axis == TIME:
                if i == 0:
                    table_str += '{:11s} |'.format('t = START')
                elif i == table.shape[0] - 1:
                    table_str += '{:11s} |'.format('t = END')
                else:
                    table_str += '{:^11s} |'.format('t = ' + str(i))
            elif y_axis == STATE:
                table_str += '{:^11s} |'.format('s = ' + str(i))
            else:
                raise TypeError('Please enter a valid y-axis type.')

            for j in range(table.shape[1]):
                table_str += '{:1s} {:4.3E} {:1s}'.format('|', table[i][j], '')
            table_str += '||\n'

        print(table_str)
