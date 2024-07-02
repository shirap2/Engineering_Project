from generate_info.gen_all import create_pdf_file
from create_input.create_input_files import Organ
import argparse


default_liver = ['E_N_', 'N_M_', 'M_I_', 'M_N_', 'G_Y_', 'S_I_', 'S_N_', 'F_Y_Ga_', 'T_N_', 'G_B_', 'C_A_', 'B_B_S_',
      'A_S_S_', 'A_S_H_', 'Z_Aa_', 'H_G_', 'A_W_', 'B_T_']

default_brain = ['SZ0', 'VA0', 'MY1', 'RL0', 'DD1', 'MG0', 'AN0', 'BY0', 'LS0', 'SM0', 'SF0', 'LS1', 'TM0', 'HD0', 'AA0', 'IM0',
      'MB0', 'YA0', 'BH0', 'RS0', 'AZ0', 'LA0', 'HS0', 'AF0', 'ZR0', 'NN0', 'NM1', 'DT0', 'HM0', 'ZI0']

default_lungs = ['M_G_', 'A_Z_A_', 'N_M_R_', 'S_I_', 'S_N_', 'F_Y_Ga_', 'G_B_', 'C_A_', 'B_S_Ya_', 'B_B_S_', 'P_I_', 'N_Na_',
      'A_S_H_', 'Z_Aa_', 'A_Y_', 'A_A_', 'G_Ea_', 'L_I_', 'M_S_']


""" runs the generation of all elements of all the dataset patients and organs 
- displays them onto a PDF for further checking """
def create_pdf_of_entire_dataset(liver, brain, lungs):
    for name in lungs:
        try:
            organ = Organ.LUNGS
            create_pdf_file(name, organ)
            print(f'{name}: SUCCESS')
        except Exception as e:
            print(f'{name}: FAILURE - {e}')

    for name in liver:
        try:
            organ = Organ.LIVER
            create_pdf_file(name, organ)
            print(f'{name}: SUCCESS')
        except Exception as e:
            print(f'{name}: FAILURE - {e}')

    for name in brain:
        try:
            organ = Organ.BRAIN
            create_pdf_file(name, organ)
            print(f'{name}: SUCCESS')
        except Exception as e:
            print(f'{name}: FAILURE - {e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate PDFs for organs from a list of patient names.')
    parser.add_argument('--lungs', nargs='*', help='List of patient names for lungs')
    parser.add_argument('--liver', nargs='*', help='List of patient names for liver')
    parser.add_argument('--brain', nargs='*', help='List of patient names for brain')
    parser.add_argument('--full_dataset', nargs='*', help='List of patient names for brain')
    args = parser.parse_args()

    lungs, liver, brain = [], [], []

    if args.full_dataset == 'True':
        lungs = default_lungs
        liver = default_liver
        brain = default_brain
    else:
        if args.lungs is not None:
            lungs = args.lungs

        if args.liver is not None:
            liver = args.liver

        if args.brain is not None:
            brain = args.brain

    # test_data_representation_elements(args.lungs, args.liver, args.brain)
    # test_data_representation_elements([], [], [])
    create_pdf_of_entire_dataset(liver, brain, lungs)