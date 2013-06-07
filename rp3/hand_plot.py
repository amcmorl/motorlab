from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, array, radians, pi, linspace, concatenate
from vectors import Rx, Ry, Rz, ypr2mat, unitvec

'''
Reference axes are:
<- x -> ; ^ y \/ ; in/out z
'''

bone_lengths = array([[43.5, 24.5, 22],
                      [43.5, 24.5, 22],
                      [43.5, 24.5, 22],
                      [43.5, 24.5, 22],
                      [33,   17.5, 22]])
ATTACHMENT_POINTS = array([[0,0,0],
                           [24,29,0],
                           [48,40,0],
                           [72,28,0],
                           [0,-65,0]])
digit_pos = { 'index'  :0,
              'middle' :1,
              'ring'   :2,
              'little' :3,
              'thumb'  :4  }
                      
#===============================================================================
# Palm class
#===============================================================================

class Palm(object):
    def __init__(self, orientation=[0,0,0]):
        '''
        Parameters
        ----------
        orientation : sequence
          (3,) yaw, pitch, roll
        '''
        # orientation is yaw, pitch, roll, but stored internally
        # as cori matrix
        self.rotation = ypr2mat(orientation)
        
        # attachment points at reference orientation
        ref_attachment_points = ATTACHMENT_POINTS            
        self.attachment_points = dot(ATTACHMENT_POINTS, self.rotation)
        
    def get_ptb(self, finger_pos):
        '''
        Parameters
        ----------
        finger_pos : int
          finger in order: 0=index,1=middle,2=ring,3=little, 4=thumb

        Returns
        -------
        ptb : ndarray
          attachment points of fingers
        '''
        return self.attachment_points[finger_pos]

    def get_global_rotation(self):
        return self.rotation

#===============================================================================
# Bone class
#===============================================================================

class Bone(object):
    def __init__(self,
                 parent=None,
                 length=1.,
                 attachment=None,
                 ref_vec=array([0,1,0])):
        '''
        Parameters
        ----------
        parent : Bone
          bone to which this one is attached
        length : scalar
          length of bone
        attachment : function
          callable to get attachment point of bone
        ref_vec : ndarray
          vector describing reference orientation of bone
          (that occurring at all zero angles)
        '''
        self.get_pta = attachment
        self.vec = unitvec(ref_vec) * length
        self.child = None
        self.parent = parent
                    
    def get_local_rotation(self):
        '''
        Override in descendents.

        Returns
        -------
        rotmat : ndarray
          rotation matrix for local rotations applied to bone from reference
          vector by joint angles
        '''
        pass

    def get_global_rotation(self):
        '''
        Total rotation is matrix product of:
        1) rotation applied to ref_vec by this joint's angles
        2) rotation of parent bone

        Returns
        -------
        rotmat : ndarray
          rotation matrix for all rotations applied to bone
          includes local rotations and those applied to parent
        '''
        local_rotation = self.get_local_rotation()
        parent_global_rotation = self.parent.get_global_rotation()
        return dot(local_rotation, parent_global_rotation)
    
    def get_ptb(self):
        '''
        Calculate pt b from vec, offset (pt a), and rotation.

        Returns
        -------
        pt : ndarray
          co-ordinates of distal end of bone
        '''        
        return dot(self.vec, self.get_global_rotation()) + self.get_pta()
        
#===============================================================================
# Proximal Phalanges class            
#===============================================================================
            
class Phalanx2D(Bone):
    def __init__(self, parent, length, attachment=None):
        '''
        Create a phalanx with a 2 degree-of-freedom joint.
        
        Parameters
        ----------
        parent : Bone
          bone to which this one is attached
        length : scalar
          length of bone
        attachment : function
          callable to get attachment point of bone
        '''
        # initialize values
        self.abduction = 0.
        self.flexion = 0.
        
        # set links
        if attachment == None:
            attachment = lambda : parent.get_ptb()
        Bone.__init__(self, parent, length, attachment=attachment)
        
    def set_angles(self, abduction=None, flexion=None):
        '''
        Set joint angles.

        Parameters
        ----------
        abduction, flexion : scalar
          joint angles in degrees
        '''
        if abduction:
            self.abduction = radians(abduction)
        if flexion:
            self.flexion = radians(flexion)
        
    def get_local_rotation(self):
        '''
        Get rotation caused to reference vector, by local joint angles. For
        this reference orientation, flexion is +ve rotation about x and
        abduction is +ve rotation about z.

        Returns
        -------
        rotmat : ndarray
          rotation matrix for local rotations applied to bone from reference
          vector by joint angles
          '''
        abd = self.abduction
        flx = self.flexion
        return dot(Rx(flx), Rz(abd))
        
#===============================================================================
# Intermediate Phalanx class        
#===============================================================================
        
class Phalanx1D(Bone):
    def __init__(self, parent, length, axial_rotation=0.):
        '''
        Create a phalanx with a 1 DOF joint.

        Parameters
        ----------
        parent : Bone
          bone to which this one is attached
        length : scalar
          length of bone
        axial_rotation : scalar
          angle to rotate bone along its axis in radians
        '''
        self.flexion = 0.
        self.ax_rot = axial_rotation
        Bone.__init__(self, parent, length, attachment=parent.get_ptb)
        
    def set_angles(self, flexion):
        '''
        Set joint angles.

        Parameters
        ----------
        flexion : scalar
          joint angles in degrees
        '''
        self.flexion = radians(flexion)

    def get_local_rotation(self):
        '''
        Get rotation caused to reference vector, by local joint angles. For
        this refence orientation, flexion is +ve rotation about x and axial
        rotation about y.

        Returns
        -------
        rotmat : ndarray
          rotation matrix for local rotations applied to bone from reference
          vector by joint angles
        '''
        return dot(Rx(self.flexion), Ry(self.ax_rot))
        
#===============================================================================
# Finger class
#===============================================================================

class Finger(object):
    def __init__(self, parent, finger_pos):
        '''
        Create a finger, a collection of three bones.

        Parameters
        ----------
        parent : Palm
          palm to which finger is attached
        finger_pos : int
          index of position of finger
          see `digit_pos` for code meanings
        '''
        self.parent = parent # this is the palm
        self.finger_pos = finger_pos
        segment_lengths = bone_lengths[finger_pos]
        self.prox_phalanx = Phalanx2D( \
            self.parent, segment_lengths[0],
            attachment=lambda : parent.get_ptb(finger_pos))
        self.int_phalanx = Phalanx1D(self.prox_phalanx, segment_lengths[1])
        self.dst_phalanx = Phalanx1D(self.int_phalanx, segment_lengths[2])
    
    def get_bones(self):
        '''
        Get a list of handles to bones in the finger.

        Returns
        -------
        bones : list of Bones
          bones in the finger
        '''
        return [self.prox_phalanx, self.int_phalanx, self.dst_phalanx]

    def set_angles_from_vec(self, vec):
        '''
        Set all angles in the finger.

        Parameters
        ----------
        vec : sequence
          shape (4,), joint angles in degrees, in order:
          MCP abd, MCP flx, PIP flx, DIP flx
        '''
        self.prox_phalanx.set_angles(vec[0], vec[1])
        self.int_phalanx.set_angles(vec[2])
        self.dst_phalanx.set_angles(vec[3])

#===============================================================================
# Metacarpal class
#===============================================================================

class Metacarpal(Phalanx2D):
    def get_local_rotation(self):
        '''
        Get rotation caused to reference vector, by local joint angles. For
        this reference orientation, flexion is +ve rotation about z axis,
        and abduction is -ve rotation about y axis. Flexion rotation is
        performed first to `vec`.

        Returns
        -------
        rotmat : ndarray
          rotation matrix for local rotations applied to bone from reference
          vector by joint angles
        '''
        abd = self.abduction
        flx = self.flexion
        return dot(Rz(-flx), Ry(abd))
            
#===============================================================================
# Thumb class
#===============================================================================

class Thumb(object):
    def __init__(self, parent):
        '''
        Create a finger, a collection of three bones.

        Parameters
        ----------
        parent : Palm
          palm to which finger is attached
        '''
        self.parent = parent # this is the palm
        self.finger_pos = digit_pos['thumb']
        segment_lengths = bone_lengths[self.finger_pos]
        self.metacarpal = Metacarpal( \
            self.parent, segment_lengths[0],
            attachment=lambda : parent.get_ptb(self.finger_pos))
        self.prox_phalanx = Phalanx1D(self.metacarpal, segment_lengths[1],
                                      axial_rotation=pi/2.)
        self.dst_phalanx = Phalanx1D(self.prox_phalanx, segment_lengths[2])

    def get_bones(self):
        '''
        Get a list of handles to bones in the finger.

        Returns
        -------
        bones : list of Bones
          bones in the finger
        '''
        return [self.metacarpal, self.prox_phalanx, self.dst_phalanx]
        
    def set_angles_from_vec(self, vec):
        '''
        Set all angles in the finger.

        Parameters
        ----------
        vec : sequence
          shape (4,), joint angles in degrees, in order:
          CMC abd, CMC flx, MCP flx, IP flx
        '''
        self.metacarpal.set_angles(vec[0], vec[1])
        self.prox_phalanx.set_angles(vec[2])
        self.dst_phalanx.set_angles(vec[3])
        
#===============================================================================
# Hand class
#===============================================================================

class Hand(object):
    def __init__(self):
        '''
        Create a hand instance. A hand consists of a palm and five digits.

        Notes
        -----
        to do : implement left and right hand versions, currently only right
        '''
        self.palm = Palm()
        self.digits = [Finger(self.palm, 0),
                       Finger(self.palm, 1),
                       Finger(self.palm, 2),
                       Finger(self.palm, 3),
                       Thumb(self.palm)]
        self.bones = []
        for digit in self.digits:
            self.bones.extend(digit.get_bones())
            
    def set_angles_from_vec(self, angles_vec):
        '''
        Set all angles in the hand from a single vector.
        
        Parameters
        ----------
        vec : sequence
          shape (20,), vector of joint angles, in order:
          4 DOF per finger: MCP abduction, MCP flexion, DIP flexion, PIP flexion
          or thumb: CMC abduction, CMC flexion, MCP flexion, IP flexion
        '''
        for i, digit in enumerate(self.digits):
            digit.set_angles_from_vec(angles_vec[4*i:4*i+4])
        
#===============================================================================
# main code
#===============================================================================

def draw_hand(hand, ax=None, **kwargs):
    '''
    Draw points of a hand, using mplot3d.

    Parameters
    ----------
    hand : Hand
      hand to draw
    ax : matplotlib.Axes object
      existing axis object to use
    '''
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for bone in hand.bones:
        pts = concatenate([bone.get_pta()[None],
                           bone.get_ptb()[None]],
                          axis=0)
        x,y,z = pts.T
        ax.plot(x,y,z, 'o', **kwargs)
        ax.set_xlim(-40,100)
        ax.set_ylim(-100,100)
        ax.set_zlim(-100,100)

def draw_closing_hand():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hand = Hand()
    angles = array([ 0, 0, 0, 0,
                     0, 0, 0, 0,
                     0, 0, 0, 0,
                     0, 0, 0, 0,
                    90, 60, 0, 0], dtype=float)
    finish = array([ 0, 60, 90, 80,
                     0, 60, 90, 80,
                     0, 60, 90, 80,
                     0, 60, 90, 80,
                    90, 60, 10, 20], dtype=float)
    line_skip = 5
    tstep = 25 # number of time steps to perform
    #hand.set_angles_from_vec(angles)
    #draw_hand(hand, ax=ax)
    step = (finish - angles) / tstep
    
    for i, a in enumerate(linspace(0.3, 1, tstep)):
        angles += step
        hand.set_angles_from_vec(angles)
        if (i % line_skip == 0) or (i == 0):
            d = {'linestyle': '-'}
        elif (i == tstep - 1):
            d = {'linestyle' : '-', 'linewidth' : 3}
        else:
            d = {}
        draw_hand(hand, ax=ax, alpha=a, **d)
        
    return hand
    
#if __name__ == "__main__":
#    hand = draw_closing_hand()
