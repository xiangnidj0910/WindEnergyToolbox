'''
Created on 20/01/2014

@author: MMPE

See documentation of HTCFile below

'''
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import int
from builtins import str
from future import standard_library
import os
from wetb.wind.shear import log_shear, power_shear
standard_library.install_aliases()



class HTCDefaults(object):


    empty_htc = """begin simulation;
        time_stop 600;
        solvertype    1;    (newmark)
        on_no_convergence continue;
        convergence_limits 1E3 1.0 1E-7; ; . to run again, changed 07/11
        begin newmark;
          deltat    0.02;
        end newmark;
    end simulation;
    ;
    ;----------------------------------------------------------------------------------------------------------------------------------------------------------------
    ;
    begin new_htc_structure;
      begin orientation;
      end orientation;
      begin constraint;
      end constraint;
    end new_htc_structure;
    ;
    ;----------------------------------------------------------------------------------------------------------------------------------------------------------------
    ;
    begin wind ;
      density                 1.225 ;
      wsp                     10   ;
      tint                    1;
      horizontal_input        1     ;            0=false, 1=true
      windfield_rotations     0 0.0 0.0 ;    yaw, tilt, rotation
      center_pos0             0 0 -30 ; hub heigth
      shear_format            1   0;0=none,1=constant,2=log,3=power,4=linear
      turb_format             0     ;  0=none, 1=mann,2=flex
      tower_shadow_method     0     ;  0=none, 1=potential flow, 2=jet
    end wind;
    ;
    ;----------------------------------------------------------------------------------------------------------------------------------------------------------------
    ;
    begin dll;
    end dll;
    ;
    ;----------------------------------------------------------------------------------------------------------------------------------------------------------------
    ;
    begin output;
      general time;
    end output;
    exit;"""


    def add_mann_turbulence(self, L=29.4, ae23=1, Gamma=3.9, seed=1001, high_frq_compensation=True,
                            filenames=None,
                            no_grid_points=(16384, 32, 32), box_dimension=(6000, 100, 100),
                            dont_scale=False,
                            std_scaling=None):
        wind = self.add_section('wind')
        wind.turb_format = 1
        mann = wind.add_section('mann')
        if 'create_turb_parameters' in mann:
            mann.create_turb_parameters.values = [L, ae23, Gamma, seed, int(high_frq_compensation)]
        else:
            mann.add_line('create_turb_parameters', [L, ae23, Gamma, seed, int(high_frq_compensation)], "L, alfaeps, gamma, seed, highfrq compensation")
        if filenames is None:
            fmt = "mann_l%.1f_ae%.2f_g%.1f_h%d_%dx%dx%d_%.3fx%.2fx%.2f_s%04d%c.turb"
            import numpy as np
            dxyz = tuple(np.array(box_dimension) / no_grid_points)
            filenames = ["./turb/" + fmt % ((L, ae23, Gamma, high_frq_compensation) + no_grid_points + dxyz + (seed, uvw)) for uvw in ['u', 'v', 'w']]
        if isinstance(filenames, str):
            filenames = ["./turb/%s_s%04d%s.bin" % (filenames, seed, c) for c in ['u', 'v', 'w']]
        for filename, c in zip(filenames, ['u', 'v', 'w']):
            setattr(mann, 'filename_%s' % c, filename)
        for c, n, dim in zip(['u', 'v', 'w'], no_grid_points, box_dimension):
            setattr(mann, 'box_dim_%s' % c, "%d %.4f" % (n, dim / (n - 1)))
        if dont_scale:
            mann.dont_scale = 1
        else:
            try:
                del mann.dont_scale
            except KeyError:
                pass
        if std_scaling is not None:
            mann.std_scaling = "%f %f %f" % std_scaling
        else:
            try:
                del mann.std_scaling
            except KeyError:
                pass
            


    def add_turb_export(self, filename="export_%s.turb", samplefrq = None):
        exp = self.wind.add_section('turb_export', allow_duplicate=True)
        for uvw in 'uvw':
            exp.add_line('filename_%s'%uvw, [filename%uvw])
        sf = samplefrq or max(1,int( self.wind.mann.box_dim_u[1]/(self.wind.wsp[0] * self.deltat())))
        exp.samplefrq = sf
        if "time" in self.output:
            exp.time_start = self.output.time[0]
        else:
            exp.time_start = 0
        exp.nsteps = (self.simulation.time_stop[0]-exp.time_start[0]) / self.deltat()
        for vw in 'vw':
            exp.add_line('box_dim_%s'%vw, self.wind.mann['box_dim_%s'%vw].values)

        

    def import_dtu_we_controller_input(self, filename):
        dtu_we_controller = [dll for dll in self.dll if dll.name[0] == 'dtu_we_controller'][0]
        with open (filename) as fid:
            lines = fid.readlines()
        K_r1 = float(lines[1].replace("K = ", '').replace("[Nm/(rad/s)^2]", ''))
        Kp_r2 = float(lines[4].replace("Kp = ", '').replace("[Nm/(rad/s)]", ''))
        Ki_r2 = float(lines[5].replace("Ki = ", '').replace("[Nm/rad]", ''))
        Kp_r3 = float(lines[7].replace("Kp = ", '').replace("[rad/(rad/s)]", ''))
        Ki_r3 = float(lines[8].replace("Ki = ", '').replace("[rad/rad]", ''))
        KK = lines[9].split("]")
        KK1 = float(KK[0].replace("K1 = ", '').replace("[deg", ''))
        KK2 = float(KK[1].replace(", K2 = ", '').replace("[deg^2", ''))
        cs = dtu_we_controller.init
        cs.constant__11.values[1] = "%.6E" % K_r1
        cs.constant__12.values[1] = "%.6E" % Kp_r2
        cs.constant__13.values[1] = "%.6E" % Ki_r2
        cs.constant__16.values[1] = "%.6E" % Kp_r3
        cs.constant__17.values[1] = "%.6E" % Ki_r3
        cs.constant__21.values[1] = "%.6E" % KK1
        cs.constant__22.values[1] = "%.6E" % KK2


class HTCExtensions(object):
    def get_shear(self):
        shear_type, parameter = self.wind.shear_format.values
        z0 = -self.wind.center_pos0[2]
        wsp = self.wind.wsp[0]
        if shear_type==1: #constant
            return lambda z : parameter
        elif shear_type==3:
            return power_shear(parameter, z0, wsp)
        else:
            raise NotImplementedError