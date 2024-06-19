from generate_info.gen_all import create_pdf_file
from create_input.create_input_files import Organ

liver = ['E_N_', 'N_M_', 'M_I_', 'M_N_', 'G_Y_', 'S_I_', 'S_N_', 'F_Y_Ga_', 'T_N_', 'G_B_', 'C_A_', 'B_B_S_',
      'A_S_S_', 'A_S_H_', 'Z_Aa_', 'H_G_', 'A_W_', 'B_T_']

brain = ['SZ0', 'VA0', 'MY1', 'RL0', 'DD1', 'MG0', 'AN0', 'BY0', 'LS0', 'SM0', 'SF0', 'LS1', 'TM0', 'HD0', 'AA0', 'IM0',
      'MB0', 'YA0', 'BH0', 'RS0', 'AZ0', 'LA0', 'HS0', 'AF0', 'ZR0', 'NN0', 'NM1', 'DT0', 'HM0', 'ZI0']

lungs = ['M_G_', 'A_Z_A_', 'N_M_R_', 'S_I_', 'S_N_', 'F_Y_Ga_', 'G_B_', 'C_A_', 'B_S_Ya_', 'B_B_S_', 'P_I_', 'N_Na_',
      'A_S_H_', 'Z_Aa_', 'A_Y_', 'A_A_', 'G_Ea_', 'L_I_', 'M_S_']


def create_pdf_of_entire_dataset():
    for name in lungs:
        try:
            organ = Organ.LUNGS
            create_pdf_file(name, organ)
        except Exception as e:
            print(f'{name}: {e}')

    for name in liver:
        try:
            organ = Organ.LIVER
            create_pdf_file(name, organ)
        except Exception as e:
            print(f'{name}: {e}')

    for name in brain:
        try:
            organ = Organ.BRAIN
            create_pdf_file(name, organ)
        except Exception as e:
            print(f'{name}: {e}')

create_pdf_of_entire_dataset()