from numpy import cos, sin, arctan2, arccos, arcsin, array, dot, \
    square, sqrt, nonzero, zeros, sum, nan_to_num, diag, real, abs, pi
from numpy.core.getlimits import finfo

#==============================================================================
# Axial rotation matrices
#==============================================================================

def Rx(theta):
    '''
    Construct rotation matrix for rotation about x.

    Parameters
    ----------
    theta : float
      angle in radians

    Returns
    -------
    rotation_matrix : ndarray
      3x3 rotation matrix
    '''
    t = theta
    return array([[1,      0,       0],
                  [0, cos(t), -sin(t)],
                  [0, sin(t),  cos(t)]])

def Ry(theta):
    '''
    Construct rotation matrix for rotation about y.

    Parameters
    ----------
    theta : float
      angle in radians

    Returns
    -------
    rotation_matrix : ndarray
      3x3 rotation matrix
    '''
    t = theta
    return array([[ cos(t), 0, sin(t)],
                  [ 0,      1,      0],
                  [-sin(t), 0, cos(t)]])

def Rz(theta):
    '''
    Construct rotation matrix for rotation about z.

    Parameters
    ----------
    theta : float
      angle in radians

    Returns
    -------
    rotation_matrix : ndarray
      3x3 rotation matrix
    '''
    t = theta
    return array([[cos(t), -sin(t), 0],
                  [sin(t),  cos(t), 0],
                  [0,            0, 1]])

# def ypr2mat(ypr):
#     '''
#     Construct rotation matrix from yaw, pitch, roll.

#     Parameters
#     ----------
#     ypr : array_like
#       shape (3,),  yaw, pitch and roll angles

#     Returns
#     -------
#     rotation_matrix : ndarray
#       shape (3,3) rotation matrix
#     '''
#     return dot(dot(Rx(ypr[0]), Ry(ypr[1])), Rz(ypr[2]))

def rotmat2rotvec(A):
    '''
    '''
    eps = 1e-8
    costheta = (diag(A).sum() - 1) / 2.
    if abs(costheta - 1.) < eps:
        costheta = 1
    theta = real(arccos(costheta))
    if (theta < eps) | (abs(pi - theta) < eps) | (abs(pi + theta) < eps):
        e = array([0, 0, 0])
    else:
        sin_theta = sin(theta)
        e1 = (A[2,1] - A[1,2]) / (2 * sin_theta)
        e2 = (A[0,2] - A[2,0]) / (2 * sin_theta)
        e3 = (A[1,0] - A[0,1]) / (2 * sin_theta)
        e = array([e1, e2, e3])
    return e, theta

#==============================================================================
# Euler (yaw-pitch-roll) specification
#==============================================================================

def eul2DCM(eul, order):
    '''
    Construct direction cosine matrix (DCM, aka rotation matrix) from euler
    angles.

    Parameters
    ----------
    eul : array_like
        shape (3,),  euler angles
    order : string
        len 3, order of rotations

    Returns
    -------
    DCM : ndarray
        shape (3,3) rotation matrix
    '''
    # I feel like this should use a decorator, but I don't really know how to do that.
    if order == 'xyx':
        return dot(dot(Rx(eul[0]), Ry(eul[1])), Rx(eul[2]))
    elif order == 'yzy':
        return dot(dot(Ry(eul[0]), Rz(eul[1])), Ry(eul[2]))
    elif order == 'zxz':
        return dot(dot(Rz(eul[0]), Rx(eul[1])), Rz(eul[2]))
    elif order == 'xzx':
        return dot(dot(Rx(eul[0]), Rz(eul[1])), Rx(eul[2]))
    elif order == 'yxy':
        return dot(dot(Ry(eul[0]), Rx(eul[1])), Ry(eul[2]))
    elif order == 'zyz':
        return dot(dot(Rz(eul[0]), Ry(eul[1])), Rz(eul[2]))
    elif order == 'xyz':
        return dot(dot(Rx(eul[0]), Ry(eul[1])), Rz(eul[2]))
    elif order == 'yzx':
        return dot(dot(Ry(eul[0]), Rz(eul[1])), Rx(eul[2]))
    elif order == 'zxy':
        return dot(dot(Rz(eul[0]), Rx(eul[1])), Ry(eul[2]))
    elif order == 'xzy':
        return dot(dot(Rx(eul[0]), Rz(eul[1])), Ry(eul[2]))
    elif order == 'yxz':
        return dot(dot(Ry(eul[0]), Rx(eul[1])), Rz(eul[2]))
    elif order == 'zyx':
        return dot(dot(Rz(eul[0]), Ry(eul[1])), Rx(eul[2]))

def eul2quat(eul, order):
    '''
    Construct rotation quaternion from euler angles.

    Parameters
    ----------
    eul : array_like
        shape (3,),  euler angles
    order : string
        len 3, order of rotations

    Returns
    -------
    quat : array_like
        shape (4,),  quaternion
    '''
    quat = zeros(4)
    c1 = cos(eul[0]/2)
    c2 = cos(eul[1]/2)
    c3 = cos(eul[2]/2)
    s1 = sin(eul[0]/2)
    s2 = sin(eul[1]/2)
    s3 = sin(eul[2]/2)
    c13 = cos((eul[0]+eul[2])/2)
    s13 = sin((eul[0]+eul[2])/2)
    c1_3 = cos((eul[0]-eul[2])/2)
    s1_3 = sin((eul[0]-eul[2])/2)
    c3_1 = cos((eul[2]-eul[0])/2)
    s3_1 = sin((eul[2]-eul[0])/2)
    if order == 'xyx':
        quat = array([c2*s13, s2*c1_3, s2*s1_3, c2*c13])
    elif order == 'yzy':
        quat = array([s2*s1_3, c2*s13, s2*c1_3, c2*c13])
    elif order == 'zxz':
        quat = array([s2*c1_3, s2*s1_3, c2*s13, c2*c13])
    elif order == 'xzx':
        quat = array([c2*s13, s2*s3_1, s2*c3_1, c2*c13])
    elif order == 'yxy':
        quat = array([s2*c3_1, c2*s13, s2*s3_1, c2*c13])
    elif order == 'zyz':
        quat = array([s2*s3_1, s2*c3_1, c2*s13, c2*c13])
    elif order == 'xyz':
        quat = array([s1*c2*c3 + c1*s2*s3,
                      c1*s2*c3-s1*c2*s3,
                      c1*c2*s3 + s1*s2*c3,
                      c1*c2*c3 - s1*s2*s3])
    elif order == 'yzx':
        quat = array([c1*c2*s3 + s1*s2*c3,
                      s1*c2*c3 + c1*s2*s3,
                      c1*s2*c3 - s1*c2*s3,
                      c1*c2*c3 - s1*s2*s3])
    elif order == 'zxy':
        quat = array([c1*s2*c3 - s1*c2*s3,
                      c1*c2*s3 + s1*s2*c3,
                      s1*c2*c3 + c1*s2*s3,
                      c1*c2*c3 - s1*s2*s3])
    elif order == 'xzy':
        quat = array([s1*c2*c3 - c1*s2*s3,
                      c1*c2*s3 - s1*s2*c3,
                      c1*s2*c3 + s1*c2*s3,
                      c1*c2*c3 + s1*s2*s3])
    elif order == 'yxz':
        quat = array([c1*s2*c3 + s1*c2*s3,
                      s1*c2*c3 - c1*s2*s3,
                      c1*c2*s3 - s1*s2*c3,
                      c1*c2*c3 + s1*s2*s3])
    elif order == 'zyx':
        quat = array([c1*c2*s3 - s1*s2*c3,
                      c1*s2*c3 + s1*c2*s3,
                      s1*c2*c3 - c1*s2*s3,
                      c1*c2*c3 + s1*s2*s3])
	
    # Normalize quaternions in case of deviation from unity
    return quat/sqrt(sum(square(quat),0))

def eul2axang(eul, order):
    '''
    Construct euler axis/angle from euler angles.

    Parameters
    ----------
    eul : array_like
        shape (3,),  yaw, pitch and roll angles
    order : string
        len 3, order of rotations

    Returns
    -------
    axang : array_like
        shape (3,),  euler axis/angle in compact form
    '''
    return quat2axang(eul2quat(eul, order))

#==============================================================================
# Axis-angle specification
#==============================================================================

def axang2quat(axang):
    '''
    Construct rotation quaternion from axis/angle.

    Parameters
    ----------
    axang : array_like
        shape (3,),  axis/angle in compact form

    Returns
    -------
    quat : array_like
        shape (4,),  quaternion
    '''
    # Extract angle and normalized axis
    axis = zeros(3)
    angle = 0
    if axang.shape[0] == 3:
        angle = sqrt(sum(square(axang),0))
        axis = axang/angle
    elif axang.shape[0] == 4:
        angle = axang[3]
        axis = axang[0:3]
    axis = nan_to_num(axis)
    quat = array([axis[0]*sin(angle/2),
                  axis[1]*sin(angle/2),
                  axis[2]*sin(angle/2),
                  cos(angle/2)])
    # Normalize quaternions in case of deviation from unity
    return quat/sqrt(sum(square(quat),0))

def axang2eul(axang, order):
    '''
    Construct euler angles from euler axis/angle.

    Parameters
    ----------
    axang : array_like
        shape (3,),  euler axis/angle in compact form
    order : string
        len 3, order of rotations

    Returns
    -------
    eul : array_like
        shape (3,),  yaw, pitch and roll angles
    '''
    return quat2eul(axang2quat(axang), order)

def axang2DCM(axang):
    '''
    Construct euler angles from euler axis/angle.

    Parameters
    ----------
    axang : array_like
        shape (3,),  euler axis/angle in compact form

    Returns
    -------
    DCM : ndarray
        shape (3,3) rotation matrix
    '''
    return quat2DCM(axang2quat(axang))

#==============================================================================
# rotation matrices
#==============================================================================

def DCM2quat(DCM):
    '''
    Construct rotation quaternion from direction cosine matrix (DCM, aka
    rotation matrix).

    Parameters
    ----------
    DCM : ndarray
        shape (3,3) rotation matrix

    Returns
    -------
    quat : array_like
        shape (4,),  quaternion
    '''
    DCM = DCM.transpose()
    quat = zeros(4)
    denom = array([0.5*sqrt(1+DCM[0,0]-DCM[1,1]-DCM[2,2]),
                   0.5*sqrt(1-DCM[0,0]+DCM[1,1]-DCM[2,2]),
                   0.5*sqrt(1-DCM[0,0]-DCM[1,1]+DCM[2,2]),
                   0.5*sqrt(1+DCM[0,0]+DCM[1,1]+DCM[2,2])])
    denommaxind = array(nonzero(denom==denom.max())).min()
    #noinspection PySimplifyBooleanCheck
    if denommaxind == 0:
        quat[0] = denom[0]
        quat[1] = (DCM[0,1]+DCM[1,0])/(4*quat[0])
        quat[2] = (DCM[0,2]+DCM[2,0])/(4*quat[0])
        quat[3] = (DCM[1,2]-DCM[2,1])/(4*quat[0])
    elif denommaxind == 1:
        quat[0] = denom[1]
        quat[1] = (DCM[0,1]+DCM[1,0])/(4*quat[1])
        quat[2] = (DCM[1,2]+DCM[2,1])/(4*quat[1])
        quat[3] = (DCM[2,0]-DCM[2,0])/(4*quat[1])
    elif denommaxind == 2:
        quat[2] = denom[2]
        quat[0] = (DCM[0,2]+DCM[2,0])/(4*quat[2])
        quat[1] = (DCM[1,2]+DCM[2,1])/(4*quat[2])
        quat[3] = (DCM[0,1]-DCM[1,0])/(4*quat[2])
    elif denommaxind == 3:
        quat[3] = denom[3]
        quat[0] = (DCM[1,2]-DCM[2,1])/(4*quat[3])
        quat[1] = (DCM[2,0]-DCM[0,2])/(4*quat[3])
        quat[2] = (DCM[0,1]-DCM[1,0])/(4*quat[3])
    # Normalize quaternions in case of deviation from unity
	quat = array(quat)
    return quat/sqrt(sum(square(quat),0))

def DCM2axang(DCM):
    '''
    Construct euler angles from euler axis/angle.

    Parameters
    ----------
    DCM : ndarray
        shape (3,3) rotation matrix

    Returns
    -------
    axang : array_like
        shape (3,),  euler axis/angle in compact form
    '''
    return quat2axang(DCM2quat(DCM))

def DCM2eul(DCM, order):
    '''
    Construct euler angles from euler axis/angle.

    Parameters
    ----------
    DCM : ndarray
        shape (3,3) rotation matrix
    order : string
        len 3, order of rotations

    Returns
    -------
    eul : array_like
        shape (3,),  yaw, pitch and roll angles
    '''
    return quat2eul(DCM2quat(DCM), order)

#==============================================================================
# quaternions
#==============================================================================

def quat2eul(quat, order):
    '''
    Construct euler angles from rotation quaternion.

    Parameters
    ----------
    quat : array_like
        shape (4,),  quaternion
    order : string
        len 3, order of rotations

    Returns
    -------
    eul : array_like
        shape (3,),  euler angles
    '''
    psi = theta = phi = 0
    if order == 'xyx':
        psi = arctan2(quat[0] * quat[1] + quat[2] * quat[3],
                    quat[1] * quat[3] - quat[0] * quat[2])
        theta = arccos(square(quat[3]) + square(quat[0]) - \
                         square(quat[1]) - square(quat[2]))
        phi = arctan2(quat[0] * quat[1] - quat[2] * quat[3],
                      quat[0] * quat[2] + quat[1] * quat[3])
#        Euler_type=2
    elif order == 'yzy':
        psi = arctan2(quat[0] * quat[3] + quat[1] * quat[2],
                    quat[2] * quat[3] - quat[0] * quat[1])
        theta = arccos(square(quat[3]) - square(quat[0]) + \
                           square(quat[1]) - square(quat[2]))
        phi=arctan2(quat[1] * quat[2] - quat[0] * quat[3],
                    quat[0] * quat[1] + quat[2] * quat[3])
#        Euler_type=2
    elif order == 'zxz':
        psi = arctan2(quat[0] * quat[2] + quat[1] * quat[3],
                      quat[0] * quat[3] - quat[1] * quat[2])
        theta = arccos(square(quat[3]) - square(quat[0]) - \
                           square(quat[1]) + square(quat[2]))
        phi=arctan2(quat[0] * quat[2] - quat[1] * quat[3],
                    quat[0] * quat[3] + quat[1] * quat[2])
#        Euler_type=2
    elif order == 'xzx':
        psi=arctan2(quat[0]*quat[2]-quat[1]*quat[3],
                    quat[0]*quat[1]+quat[2]*quat[3])
        theta=arccos(square(quat[3])+square(quat[0])- \
                         square(quat[1])-square(quat[2]))
        phi=arctan2(quat[0]*quat[2]+quat[1]*quat[3],
                    quat[2]*quat[3]-quat[0]*quat[1])
#        Euler_type=2
    elif order == 'yxy':
        psi=arctan2(quat[0]*quat[1]-quat[2]*quat[3],
                    quat[0]*quat[3]+quat[1]*quat[2])
        theta=arccos(square(quat[3])-square(quat[0])+ \
                         square(quat[1])-square(quat[2]))
        phi=arctan2(quat[0]*quat[1]+quat[2]*quat[3],
                    quat[0]*quat[3]-quat[1]*quat[2])
#        Euler_type=2
    elif order == 'zyz':
        psi=arctan2(quat[1] * quat[2] - quat[0] * quat[3],
                    quat[0] * quat[2] + quat[1] * quat[3])
        theta=arccos(square(quat[3])-square(quat[0]) - \
                         square(quat[1])+square(quat[2]))
        phi=arctan2(quat[0]*quat[3]+quat[1]*quat[2],
                    quat[1]*quat[3]-quat[0]*quat[2])
#        Euler_type=2
    elif order == 'xyz':
        psi=arctan2(2*(quat[0]*quat[3]-quat[1]*quat[2]),
                    square(quat[3])-square(quat[0]) - \
                        square(quat[1])+square(quat[2]))
        theta=arcsin(2*(quat[0]*quat[2]+quat[1]*quat[3]))
        phi=arctan2(2*(quat[2]*quat[3]-quat[0]*quat[1]),
                    square(quat[3])+square(quat[0]) - \
                        square(quat[1])-square(quat[2]))
#        Euler_type=1
    elif order == 'yzx':
        psi=arctan2(2*(quat[1]*quat[3]-quat[0]*quat[2]),
                    square(quat[3])-square(quat[0]) - \
                        square(quat[1])+square(quat[2]))
        theta=arcsin(2*(quat[0]*quat[1]+quat[2]*quat[3]))
        phi=arctan2(2*(quat[0]*quat[3]-quat[2]*quat[1]),
                    square(quat[3])+square(quat[0]) - \
                        square(quat[1])-square(quat[2]))
#        Euler_type=1
    elif order == 'zxy':
        psi=arctan2(2*(quat[2]*quat[3]-quat[0]*quat[1]),
                    square(quat[3])-square(quat[0]) - \
                        square(quat[1])+square(quat[2]))
        theta=arcsin(2*(quat[0]*quat[3]+quat[1]*quat[2]))
        phi=arctan2(2*(quat[1]*quat[3]-quat[2]*quat[0]),
                    square(quat[3])+square(quat[0]) - \
                        square(quat[1])-square(quat[2]))
#        Euler_type=1
    elif order == 'xzy':
        psi=arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),
                    square(quat[3])-square(quat[0]) + \
                        square(quat[1])-square(quat[2]))
        theta=arcsin(2*(quat[2]*quat[3]-quat[0]*quat[1]))
        phi=arctan2(2*(quat[0]*quat[2]+quat[1]*quat[3]),
                    square(quat[3])+square(quat[0]) - \
                        square(quat[1])-square(quat[2]))
#        Euler_type=1
    elif order == 'yxz':
        psi=arctan2(2*(quat[0]*quat[2]+quat[1]*quat[3]),
                    square(quat[3])-square(quat[0]) - \
                        square(quat[1])+square(quat[2]))
        theta=arcsin(2*(quat[0]*quat[3]-quat[1]*quat[2]))
        phi=arctan2(2*(quat[0]*quat[1]+quat[2]*quat[3]),
                    square(quat[3])-square(quat[0]) + \
                        square(quat[1])-square(quat[2]))
#        Euler_type=1
    elif order == 'zyx':
        psi = arctan2(2*(quat[0]*quat[1]+quat[2]*quat[3]),
                      square(quat[3])+square(quat[0]) - \
                          square(quat[1])-square(quat[2]))
        theta = arcsin(2*(quat[1] * quat[3] - quat[0] * quat[2]))
        phi = arctan2(2*(quat[0]*quat[3]+quat[2]*quat[1]),
                    square(quat[3]) - square(quat[0])- \
                        square(quat[1]) + square(quat[2]))
#        Euler_type=1

        # OUTPUT=mod([psi,theta,phi]*180/pi,360);  %deg
        # if Euler_type==1,
        # 	        sing_chk=find(abs(theta)*180/pi>89.9);
        # 	        sing_chk=sort(sing_chk(sing_chk>0));
        # 	        if size(sing_chk,1)>=1,
        # 		        error('Error: Input rotation #%s resides too close to Type 1 Euler singularity.\nType 1 Euler singularity occurs when second angle is -90 or 90 degrees.\nPlease choose different output type.',num2str(sing_chk(1,1)));
        # 	        end
        # elseif Euler_type==2,
        # 	        sing_chk=[find(abs(theta*180/pi)<0.1);find(abs(theta*180/pi-180)<0.1);find(abs(theta*180/pi-360))<0.1];
        # 	        sing_chk=sort(sing_chk(sing_chk>0));
        # 	        if size(sing_chk,1)>=1,
        # 		        error('Error: Input rotation #%s resides too close to Type 2 Euler singularity.\nType 2 Euler singularity occurs when second angle is 0 or 180 degrees.\nPlease choose different output type.',num2str(sing_chk(1,1)));
        # 	        end
        # end		
    return array([psi,theta,phi])

def quat2axang(quat):
    '''
    Construct axis/angle from rotation quaternion.

    Parameters
    -------
    quat : array_like
        shape (4,),  quaternion

    Returns
    ----------
    axang : array_like
        shape (3,),  axis/angle in compact form
    '''
    angle = 2*arctan2(sqrt(sum(quat[0:3]*quat[0:3],0)), quat[3])
    if sin(angle/2)>0:
        axis = array([quat[0]/sin(angle/2),
                      quat[1]/sin(angle/2),
                      quat[2]/sin(angle/2)])
    else:
        axis = array([1, 0, 0])
    return axis*angle

def quat2DCM(quat):
    '''
    Construct direction cosine matrix (DCM, aka rotation matrix) from rotation
    quaternion.

    Parameters
    ----------
    quat : array_like
        shape (4,),  quaternion

    Returns
    -------
    DCM : ndarray
        shape (3,3) rotation matrix
    '''
    DCM = array([[square(quat[0]) - square(quat[1]) - \
                      square(quat[2]) + square(quat[3]),
                  2 * (quat[0] * quat[1] + quat[2] * quat[3]),
                  2 * (quat[0] * quat[2] - quat[1] * quat[3])],
                 [2 * (quat[0] * quat[1] - quat[2] * quat[3]),
                  -square(quat[0]) + square(quat[1]) - \
                      square(quat[2]) + square(quat[3]),
                  2 * (quat[1] * quat[2] + quat[0] * quat[3])],
                 [2 * (quat[0] * quat[2] + quat[1] * quat[3]),
                  2 * (quat[1] * quat[2] - quat[0] * quat[3]),
                  -square(quat[0]) - square(quat[1]) + \
                      square(quat[2]) + square(quat[3])]])
    return DCM.transpose()
#    return DCM

#==============================================================================
# Rotation operations
#==============================================================================

def DCMdiff(a, b):
    '''
    Returns cartesian orientation matrix c = a - b.
    '''
    return dot(a, b.T)

