import numpy as np
import types
import vectors
from enthought.mayavi import mlab
p3d = mlab.pipeline
shoulder_length = 0.2

# ----------------------------------------------------------------------------

class Bone():
    '''
    Provides functionality common to all bones - defining some properties,
    drawing and erasing. Must be inherited from in order to be useful.
    '''

    def __init__(self,
                 attachment = None,
                 length = 1.,
                 parent = None):
        self.point_a = attachment
        self.length = length
        self.graphic_sources = []
        self.child = None
        self.parent = parent
        self.rotations = []
        if parent != None:
            parent.child = self
        self.draw()

    def draw(self):
        '''Draws arm tube in current mlab figure, and adds source to
        graphic_sources attribute.'''
        pta = self.point_a()
        ptb = self.calc_point_b()
        pts = np.vstack((pta, ptb)).transpose()
        lsrc = p3d.line_source(pts[0], pts[1], pts[2])
        self.graphic_sources.append(lsrc)
        tub = p3d.tube(lsrc)
        tub.filter.number_of_sides = 12
        tub.filter.capping = True
        surf = p3d.surface(tub)
        surf.actor.property.color = self.color

    def update(self):
        # assumes bone itself is graphic_sources[0]
        src = self.graphic_sources[0]
        src.data.points[0] = self.point_a()
        src.data.points[1] = self.calc_point_b()
        src.update()
        if self.child != None:
            self.child.update()

    def erase(self):
        '''Removes arm tube from visualization, and removes corresponding
        source reference.'''
        while self.graphic_sources:
            src = self.graphic_sources.pop()
            src.remove()

# ----------------------------------------------------------------------------

def shoulder_position():
    '''
    Returns
    -------
    position : array_like, shape (3,) = (0., 0., 0.)
        position of joint in world co-ordinates'''
    return np.array((-shoulder_length, 0., 0.))
        
class ShoulderBone(Bone):
    '''
    Static bone representing shoulder. Attachment point for humerus is
    set to be (0,0,0).
    '''

    def __init__(self,
                 attachment = shoulder_position,
                 length = shoulder_length):
        self.color = (0., 1., 0.)
        Bone.__init__(self, attachment, length)

    def calc_point_b(self):
        point_b = np.array((0., 0., 0.))
        return point_b

# ----------------------------------------------------------------------------

class HumerusBone(Bone):

    external_rotation0 = np.pi/2.
    abduction0 = 0.

    def __init__(self,
                 parent = None,
                 length = 1.,
                 abduction = abduction0,
                 external_rotation = external_rotation0):
        self.abduction = abduction
        self.external_rotation = external_rotation
        self.color = (1., 0., 0.)
        Bone.__init__(self,
                      attachment=parent.calc_point_b,
                      length=length,
                      parent=parent)
        self.rotations.append(self.abduct)
        self.rotations.append(self.external_rotate)

    # effectors
    def abduct(self, incr):
        self.abduction += incr

    def external_rotate(self, incr):
        self.external_rotation += incr

    def reset(self):
        self.external_rotation = self.external_rotation0
        self.abduction = self.abduction0
        self.update()

    # calculations
    def calc_point_b(self):
        abd = self.abduction
        rot = self.external_rotation
        point_b = self.point_a() + \
                  self.length * np.array((np.sin(abd) * np.cos(rot),
                                          np.sin(abd) * np.sin(rot),
                                          np.cos(abd)))
        return point_b

    def calc_orientation(self):
        '''Orientation is rotated by external_rotation,
        and perpendicular to bone'''
        abd = self.abduction
        rot = self.external_rotation
        orientation = np.array((np.cos(abd) * np.cos(rot),
                                np.cos(abd) * np.sin(rot),
                                -np.sin(abd)))
        return orientation

    def calc_direction(self):
        '''Vector along bone.'''
        pta = self.point_a()
        ptb = self.calc_point_b()
        return ptb - pta        

    # visualization
    def draw(self):
        Bone.draw(self)

        # draw orientation
        orient = self.calc_orientation()
        pt_center = (self.point_a() + self.calc_point_b()) / 2.
        vscat = p3d.vector_scatter(pt_center[0], pt_center[1], pt_center[2],
                                   orient[0], orient[1], orient[2])
        self.graphic_sources.append(vscat)
        g = p3d.glyph(vscat)
        g.glyph.glyph.scale_factor = 0.25
        g.actor.property.color = self.color
        g.glyph.glyph_source.glyph_position = 'tail'
        g.glyph.glyph_source.glyph_source = g.glyph.glyph_source.glyph_list[1]

    def update(self):
        Bone.update(self)
        # assume graphic_sources[1] is arrow
        src = self.graphic_sources[1]
        src.data.points = [(self.point_a() + self.calc_point_b()) / 2.]
        src.data.point_data.vectors = [self.calc_orientation()]
        src.data.point_data.vectors.name = 'vectors'
        src.update()

# ----------------------------------------------------------------------------

class UlnaBone(Bone):

    flexion0 = np.pi/2.

    def __init__(self,
                 parent = None,
                 length = 1.,
                 flexion = np.pi/2.):
        self.flexion = flexion
        self.color = (0., 0., 1.)
        Bone.__init__(self,
                      attachment=parent.calc_point_b,
                      length=length,
                      parent=parent)
        self.rotations.append(self.flex)

    def flex(self, incr):
        self.flexion += incr

    def reset(self):
        self.flexion = self.flexion0
        self.update()
        
    def calc_point_b(self):
        par_dir = self.parent.calc_direction()
        par_orient = self.parent.calc_orientation()
        thru_elbow_vector = np.cross(par_dir, par_orient)

        # now need to rotate par_dir around
        # thru_elbow_vector to get ulna direction
        ulna_vec = vectors.rotate_about_origin_3d(par_dir, thru_elbow_vector,
                                                  self.flexion)
        point_b = self.point_a() + self.length * ulna_vec
        
        #point_b = np.array((1,0,1))
        return point_b

# ----------------------------------------------------------------------------

def combine_effectors(bones):
    effectors = []
    for bone in bones:
        for effector in bone.rotations:
            effectors.append(effector)
    return effectors

class Arm():

    def __init__(self):
        self.shoulder = ShoulderBone()
        self.humerus = HumerusBone(parent=self.shoulder)
        self.ulna = UlnaBone(parent=self.humerus)
        self.bones = [self.shoulder, self.humerus, self.ulna]
        self.effectors = combine_effectors(self.bones)

    def update(self):
        self.humerus.update()

    def reset(self):
        self.humerus.reset()
        self.ulna.reset()

    def set_angles(self, external_rotation, abduction, flexion):
        self.humerus.abduction = abduction
        self.humerus.external_rotation = external_rotation
        self.ulna.flexion = flexion
        self.update()

    def get_hand_position(self):
        return self.ulna.calc_point_b()

# ----------------------------------------------------------------------------

class Neuron():
    '''
    Firing units with weighted connections to each of the six effectors:
    1) humerus.external_rotation +/-
    2) humerus.abduction +/-
    3) ulna.flexion +/-
    '''
    def __init__(self, arm, weights='random'):
        # need to register each effector (+ and -) to each cell
        self.arm = arm
        if weights == 'random':
            self.weights = np.random.normal(size=len(self.arm.effectors))
        
    def spike(self, scale_factor=0.01):
        for weight, effector in zip(self.weights, self.arm.effectors):
            effector(weight * scale_factor)
        self.arm.update()

    def fire_spike_train(self, n=100, scale_factor=0.01):
        for i in xrange(n):
            self.spike()

# ----------------------------------------------------------------------------

def calc_pd_field_one_neuron(arm, neuron, n_per_side=3):
    # pds defined by doing small delta at different points in the workspace
    # creates a 3d pd field
    # needs an co-ordinate space to joint-space conversion to get positions
    scale_factor = 0.2
    offset = np.array((0., 1., 0.5))
    inds = np.mgrid[0:n_per_side, 0:n_per_side, 0:n_per_side].astype(float)
    inds *= scale_factor
    inds += offset[:,np.newaxis,np.newaxis,np.newaxis]
    inds_flat = inds.view().reshape(3, n_per_side**3)
    pds = np.zeros((3, n_per_side, n_per_side, n_per_side))
    pds_flat = pds.view().reshape(3, n_per_side**3)
    for i, pt in enumerate(inds_flat.T):
        print pt,
        joint_angles = convert_cartesian_to_joint_space(pt, arm)
        arm.set_angles(*joint_angles)
        start_pos = arm.get_hand_position()
        neuron.spike()
        end_pos = arm.get_hand_position()
        delta = end_pos - pt
        print delta
        pds_flat.T[i] = delta
    return inds_flat, pds_flat

def draw_pd_field(pos, pds):
    x,y,z = pos
    u,v,w = pds
    g = mlab.quiver3d(x, y, z, u, v, w)
    g.glyph.glyph.scale_factor = 0.25

def convert_cartesian_to_joint_space(xyz, arm):
    xyz = np.asarray(xyz)
    x,y,z = xyz
    l_h = arm.humerus.length
    l_u = arm.ulna.length
    l_xyz = np.sqrt(np.sum(xyz**2))
    #print "Lengths:", l_h, l_u, l_xyz
    theta_flex = np.pi \
                 - np.arccos( (l_h**2 + l_u**2 - l_xyz**2) / (2 * l_h * l_u) )
    theta_hand_shoulder_elbow = np.arcsin( l_u * np.sin( theta_flex ) / l_xyz )
    theta_horiz_shoulder_hand = np.arcsin( z / l_xyz )
    theta_abduct = np.pi/2. - (theta_hand_shoulder_elbow 
                   + theta_horiz_shoulder_hand)
    theta_ext_rot = np.pi/2. \
                    - np.arcsin(x / (l_xyz * np.cos(theta_horiz_shoulder_hand)))
    return theta_ext_rot, theta_abduct, theta_flex
    
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    pass

def setup():
    sc = mlab.figure(0)
    mlab.view(90., -90., 3.)
    mlab.roll(90.)
    arm = Arm()
    neuron = Neuron(arm)
    return neuron, arm

def run(neuron, arm):
    for i in xrange(0, 1000, 10):
        neuron.spike()

def reset_view():
    mlab.view(90., -90., 3.)
    mlab.roll(90.)
