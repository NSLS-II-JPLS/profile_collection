import numpy as np
import pandas as pds
import os
import time

from ophyd import PseudoPositioner, PseudoSingle, EpicsMotor, EpicsSignal
import bluesky.preprocessors as bpp
from ophyd.pseudopos import pseudo_position_argument, real_position_argument

print(f'Loading {__file__}')


class AbsShutter(PseudoPositioner):
    # angles
    pos = Cpt(PseudoSingle, "", kind="hinted")

    # input motors
    mtr = Cpt(EpicsMotor, "", doc="The absorber slider")

    @pseudo_position_argument
    def forward(self, pseudo_pos):
        """Calculate a RealPosition from a given PseudoPosition
           based on the s_motors method

        """
        return self.RealPosition(mtr=np.ceil(pseudo_pos.pos))

    @real_position_argument
    def inverse(self, real_pos):
        """Calculate a PseudoPosition from a given RealPosition

        Parameters
        ----------
        real_position : RealPosition
            The real position input

        Returns
        -------
        pseudo_pos : PseudoPosition
            The pseudo position output
        """
        return self.PseudoPosition(pos=int(np.round(real_pos.mtr)))

#shutter = AbsShutter('XF:12ID1-ES{Slt1-Ax:X}Mtr', name='shutter')

import functools

@functools.wraps(bps.one_nd_step)
def shutter_flash_scan(*args, **kwargs):
    def cleanup_plan():
        yield from bps.mov(shutter, 0)
        yield from bps.sleep(.2)

    def collect_plan(detectors, step, pos_cache):
        motors = step.keys()
        yield from move_per_step(step, pos_cache)
        yield from bps.mov(shutter, 1)
        yield from bps.sleep(.2)
        yield from trigger_and_read(list(detectors) + list(motors))


    yield from bpp.finalize_wrapper(
        collect_plan(*args, **kwargs),
        cleanup_plan()
    )


shutter = EpicsSignal("XF:12ID1-ECAT:EL2124-00-DO1", name="shutter")


def att_setup():
    """
    Defining the PLS attenuator system. Each attenuators is defined by a thickness, a material and a name

    Returns
    -------
    att_thi: A list containing all the attenuator thicknesses
    att_mat: A list containing all the attenuator materials
    att_name: A list containing all the attenuator names
    att_name: A list of float containing the position of the attenuator motor at the beamline
    """

    # att_thi = [0, 24.28, 49.89, 74.97, 100.4, 126.11, 151.61, 177.62]
    att_thi = [0, 25.0, 50.72, 76.24, 100.76, 126.41, 152.45, 177.62]

    att_mat = ['Mo', 'Mo', 'Mo', 'Mo', 'Mo', 'Mo', 'Mo', 'Mo']
    att_name = ['att0', 'att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7']
    att_pos = [0.22, 1, 2, 3, 4, 5, 6, 7]

    '''
    # Create a dictionary of attenuators for bar1, it can only be use one att at a time
    att_bar1 = {'att0':{name:'att0', att_mat:'Mo', thickness:0.0,    postion:0.22, att_ener:1, att_mat_value:1},
                'att1':{name:'att1', att_mat:'Mo', thickness:25.0,   postion:1.0,  att_ener:1, att_mat_value:1},
                'att2':{name:'att2', att_mat:'Mo', thickness:50.72,  postion:2.0,  att_ener:1, att_mat_value:1},
                'att3':{name:'att3', att_mat:'Mo', thickness:76.24,  postion:3.0,  att_ener:1, att_mat_value:1},
                'att4':{name:'att4', att_mat:'Mo', thickness:100.76, postion:4.0,  att_ener:1, att_mat_value:1},
                'att5':{name:'att5', att_mat:'Mo', thickness:126.41, postion:5.0,  att_ener:1, att_mat_value:1},
                'att6':{name:'att6', att_mat:'Mo', thickness:152.45, postion:6.0,  att_ener:1, att_mat_value:1},
                'att7':{name:'att7', att_mat:'Mo', thickness:177.62, postion:7.0,  att_ener:1, att_mat_value:1},
                }
    
    Create a dictionary for the 2nd set of att that can be inserted in parallel
    att_bar2 = {'att0':{'name':'att0', att_mat:'Mo', 'thickness':50.0, 'position':1, 'att_ener':1, 'att_mat_value':1},
                'att1':{'name':'att1', att_mat:'Mo', 'thickness':50.0, 'position':1, 'att_ener':1, 'att_mat_value':1},
                'att2':{'name':'att2', att_mat:'Mo', 'thickness':50.0, 'position':1, 'att_ener':1, 'att_mat_value':1},
                'att3':{'name':'att3', att_mat:'Mo', 'thickness':50.0, 'position':1, 'att_ener':1, 'att_mat_value':1},
                }
    
    return att_bar1, att_bar2
    '''


    return att_thi, att_mat, att_name, att_pos

def attenuation_interpolation(path, filename, energy):
    """
    Interpolate the attenuation of a material at a given energy from the 1D curve obtained from CXRO files

    Parameters:
    -----------
    :param path: A string liking the path towards the saved txt data containing CXRO files
    :type path: string
    :param filename: A string with the filename under which the CXRO data is saved
    :type filename: string
    :param energy: the energy at which the absorption needs to be interpolated
    :type energy: float

    Returns
    -------
    att_ener: A float with the corresponding attenuation at a specific energy
    """

    array = np.loadtxt(os.path.join(path, filename))
    energy_cxro, attenuation_cxro = array[:, 0], array[:, 1]
    att_ener = np.interp(energy, energy_cxro, attenuation_cxro)
    return att_ener


def calc_attenuation(att_mat, energy):
    """
    Calculate the attenuation value of all PPLS absorbers given it material and energy. This part mainly check if the
    absorbers materials is defined and the data available, and if the picked energy is suitable for the beamline

    Parameters:
    -----------
    :param att_mat: A list containing all the attenuator materials
    :type att_mat: List of string
    :param energy: the energy at which the absorption needs to be interpolated
    :type energy: float

    Returns
    -------
    att_mat_value: A list of float which are the attenuation coefficient of each attenuators
    """
    if energy is None:
        raise attfuncs_Exception("error: No energy is input")
    elif energy > 24000 or energy < 5000:
        raise ValueError("error: energy entered is lower than 5keV or higher than 20keV")
    # else:
    #     print('The energy is', energy, 'eV')

    #load absorption of each material:
    for mat in list(set(att_mat)):
        if mat != 'Mo':
            raise ValueError("error: Foil material is not Mo and is not defined")
        else:
            att_ener_mo = attenuation_interpolation(path, file_absorption, energy)

    att_mat_value = [] * len(att_mat)
    for mat in att_mat:
        if mat == 'Mo':
            att_mat_value += [att_ener_mo]
        else:
            raise ValueError("error: No other material than Mo aare defined so far")

    return att_mat_value


def calculate_att_comb(att_thi, att_mat, energy):
    """
    Calculate all the combination of attenuation at PPLS beamline

    Parameters:
    -----------
    :param att_thi: A list containing all the attenuator thicknesses
    :type att_thi: List of string
    :param att_mat: A list containing all the attenuator materials
    :type att_mat: List of string
    :param energy: the energy at which the absorption needs to be interpolated
    :type energy: float

    Returns
    -------
    T_coefs: A list of float which contains all the transmission coefficient of all the foils at a given energy
    """
    # Calculate the absorption coefficient of every foils
    att_mat_value = calc_attenuation(att_mat, energy)
   
    # Define all the combination of foils
    t_coefs = np.zeros(np.shape(att_thi))

    # Calculate the absorption of every attenuators for a given energy and store them as a list
    for i, (att_mat_value, att_thi) in enumerate(zip(att_mat_value, att_thi)):
        t_coefs[i] = np.exp(-1 * att_thi / att_mat_value)
    return t_coefs


# ToDo: Check the T_target to constrain that it only consider T lower than a number
def best_att(T_target, energy):
    """
    Find the absorber combination that give the closest x-ray transmission from a taget value

    Parameters:
    -----------
    :param T_target: The target transmission value (has to be 1 or lower)
    :type T_target: float
    :param energy: the energy at which the absorption needs to be interpolated
    :type energy: float

    Returns
    -------
    best_att: A integer which contains the attenuator that best match the targeted transmission
    T[best_att]: A float which contains the corresponding best transmission value
    att2_name[best_att]: A string which contains the name of the matching attenuator
    att_pos[best_att]: A float which contains the physical position of the matching attenuator
    """
    att2_thi, att2_mat, att2_name, att_pos = att_setup()
    T = calculate_att_comb(att2_thi, att2_mat, energy)
    
    if T_target <= 0.99:
        # valid_att = np.where(T <= T_target)[0]
        valid_att = np.where(T)
    else:
        valid_att = np.where(T)

    best_att = np.argmin(abs(T[valid_att]-T_target))
    # print('The required attenuation is %s the best match is %s'%(T_target, T[best_att]))

    return best_att, T[best_att], att2_name[best_att], att_pos[best_att]


def put_default_absorbers(energy, default_attenuation=1e-07):
    """
    Put a default absorbers to protect the detector. With a default_attenuation factor of 1e-07,
    the full direct beam is attenuated enough to not damage the detector. This is used to set a default
    attenuator during a pre-count to define what is the necessary number of attenuators for a full image.

    Parameters:
    -----------
    :param energy: the energy at which the absorption needs to be interpolated
    :type energy: float
    :param default_attenuation: The default target transmission value (1e-07)
    :type default_attenuation: float

    Returnsput_default_absorbers
    -------
    default_factor: A float corresponding to the attenuation of the default attenuator
    """
    default_att, default_factor, default_name, default_position = best_att(default_attenuation, energy)

    if energy < 9000 and energy > 10000:
        raise ValueError('error: the energy entered is not correct')

    yield from bps.mv(abs2, default_position)
    yield from bps.sleep(1)

    return default_factor


def calculate_and_set_absorbers(energy, i_max, att, precount_time=0.1):
    """
    Define the good attenuation needed based on the maximum counts recorded by the detector on an
    image taken with a known safe number of attenuator (default attenuations). The process of 'precount'
    is used to protect the detectors

    Parameters:
    -----------
    :param energy: the energy at which the absorption needs to be interpolated
    :type energy: float
    :param i_max: The maximum counts recorded by the detector during the pre-count
    :type i_max: float
    :param att: The attenuation used during the pre-count
    :type att: float
    :param precount_time: The pre-count exposure time
    :type precount_time: float

    Returns
    -------
    best_at: An integer which contains the attenuator that best match the targeted transmission
    att_factor: A float which contains the corresponding best transmission value
    att_name: A string which contains the name of the matching attenuator
    """
    # i_max_det is the maximum allowed pixel count (nominal value is 1e4)
    i_max_det = 10000  # 50000 if lambda, 500000 if pilatus

    # Theoretical maximum count to be seen by the detector
    max_theo_precount = i_max / (att * precount_time)

    # The required attenuation to not saturate the detector, i.e. keep i_max < max_theo
    att_needed = 1 / (max_theo_precount / i_max_det)

    # Calculate and move the absorber to the chosen one
    best_at, att_factor, att_name, att_pos = best_att(att_needed, energy)
    yield from bps.mv(abs2, att_pos)

    return best_at, att_factor, att_name



all_area_dets = [lambda_det, quadem]
@bpp.stage_decorator(all_area_dets)

def define_all_att_thickness():
    ratios = np.zeros((6,))
    base_md = {'plan_name': 'calibration_att'}
    yield from bps.mv(S2.vg, 0.05)
    yield from bps.mv(S2.vc, 0)
    yield from bps.mv(S2.hg, 0.3)
    yield from bps.mv(S2.hc, 0.25)

    yield from bps.open_run(md=base_md)
    for nu in range(0, 8, 1)[::-1]:
        ratio = yield from define_att_thickness(attenuator1=nu+1, attenuator2=nu, th_angle=0.15)
        ratios[nu] = 1
        ratios *= ratio
        print('ratio between %s and %s'%(nu+1, nu), ratios[5])
    yield from bps.close_run()

    att_ener = attenuation_interpolation(path=path, filename=file_absorption, energy=energy.energy.position)
    att_thickness_new = att_ener * np.log(ratios)


    print('The new attenuators thickness are to {:.2f} um while the old were {:.2f} um'.format(att_thickness_new, att_thickness_old))
    response = input('Do you want to use the new thicknesses? (y/[n]) ')

    if response is 'y' or response is 'Y':
        print('The new attenuators thickness are to {:.2f} um'.format(att_thickness_new))
        #Need to implement the new thickness

    else:
        print('The old attenuators thicknesses were kept at {:.2f} um'.format(current_att_thickness))


#ToDo: Need to create the csv file in the startup collection of PLS
#ToDo: Make sure that the values are read and write correctly
#ToDo: Check the function name since copy and paste from SMI
#ToDo: Look if read by default
#ToDo: Need to read the attenuator thickness from the old or new calculated values
def attenuator_thickness_load():
    '''
    Load the previous attenuator thickness
    '''
    OPLS_CONFIG_FILENAME = os.path.join(get_ipython().profile_dir.location,
                                       'OPLS_attenuator_thickness.csv')
    # collect the current positions of motors
    smi_config = pds.read_csv(OPLS_CONFIG_FILENAME, index_col=0)

    att_thickness = np.zeros((8,))

    att_thickness[0] = smi_config.att0.values[-1]
    att_thickness[1] = smi_config.att1.values[-1]
    att_thickness[2] = smi_config.att2.values[-1]
    att_thickness[3] = smi_config.att3.values[-1]
    att_thickness[4] = smi_config.att4.values[-1]
    att_thickness[5] = smi_config.att5.values[-1]
    att_thickness[6] = smi_config.att6.values[-1]
    att_thickness[7] = smi_config.att7.values[-1]

    return att_thickness


current_att_thickness = attenuator_thickness_load()


def attenuator_thickness_save(attenuators_thckness):
    '''
    Save the new attenuators thckness
    '''

    SMI_CONFIG_FILENAME = os.path.join(get_ipython().profile_dir.location,
                                       'OPLS_attenuator_thickness.csv')

    # collect the current positions of motors
    current_config = {
        'att0': attenuators_thckness[0],
        'att1': attenuators_thckness[1],
        'att2': attenuators_thckness[2],
        'att3': attenuators_thckness[3],
        'att4': attenuators_thckness[4],
        'att5': attenuators_thckness[5],
        'att6': attenuators_thckness[6],
        'att7': attenuators_thckness[7],
        'time': time.ctime()
    }

    current_config_DF = pds.DataFrame(data=current_config, index=[1])

    # load the previous config file
    smi_config = pds.read_csv(SMI_CONFIG_FILENAME, index_col=0)
    smi_config_update = smi_config.append(current_config_DF, ignore_index=True)

    # save to file
    smi_config_update.to_csv(SMI_CONFIG_FILENAME)
    global current_att_thickness
    current_att_thickness = attenuator_thickness_load()
    print(current_att_thickness)


def define_att_thickness(attenuator1, attenuator2, th_angle):
    detector='lambda_det'
    attenuator_name_signal = Signal(name='attenuator_name', value='abs1')
    exposure_time = Signal(name='exposure_time', value=1)

    yield from mabt(th_angle, th_angle, 0)
    yield from bps.mv(abs2, attenuator1)
    yield from bps.sleep(10)

    yield from det_exposure_time(5, 5)
    yield from bps.mv(exposure_time, 5)
    yield from bps.mv(attenuator_name_signal, 'abs%s'%attenuator1)

    yield from bps.mv(shutter, 1)
    ret = yield from bps.trigger_and_read(area_dets, name='precount')

    # ret = yield from bps.trigger_and_read(all_area_dets +
    #                                 [geo] +
    #                                 [attenuation_factor_signal] +
    #                                 [attenuator_name_signal] +
    #                                 [exposure_time],
    #                                 name='primary')
    
    yield from bps.mv(shutter, 0)

    i_max1 = ret['%s_stats4_total'%detector]['value']


    yield from bps.mv(abs2, attenuator2)
    yield from bps.sleep(10)

    yield from det_exposure_time(5, 5)
    yield from bps.mv(exposure_time, 5)
    yield from bps.mv(attenuator_name_signal, 'abs%s'%attenuator2)

    yield from bps.mv(shutter, 1)
    ret = yield from bps.trigger_and_read(area_dets, name='precount')

    # ret = yield from bps.trigger_and_read(all_area_dets +
    #                                 [geo] +
    #                                 [attenuation_factor_signal] +
    #                                 [attenuator_name_signal] +
    #                                 [exposure_time],
    #                                 name='primary')
    yield from bps.mv(shutter, 0)
    i_max2 = ret['%s_stats4_total'%detector]['value']

    ratio = i_max2 / i_max1
    
    # att_ener = attenuation_interpolation(path, filename, energy.energy.position)

    return ratio




path = '/home/xf12id1/Downloads/'
file_absorption = 'Mo_absorption.txt'
