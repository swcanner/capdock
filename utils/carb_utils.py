import os
import numpy as np
import pandas as pd
import copy
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation as Rot
import math
from pyrosetta import *

#A = mono(name,coor,atom_names)
class mono():
    """
    Class object for a MONOMER of a carbohydrate

    Args:
        name (str): name of the residue
        coors (arr nx3): coordinates of heavy atoms
        atom_names (arr str): names of the atoms

    Variables:
        name, coor, atom_names
        atom_onehot (arr nx6 ): One-hot encoding of atoms
                    O,C,N,S,P,other
                    O first because of atom "OCN" that is O acetyl attached to C and N
        adj_mat (nxn): One-hot of bonded atoms
        edges (nx?): array of arrays of the non-sparse edge connections
        ring_atom (arr 5x1 or 6x1): defines which atoms are in the ring
        normal (arr 3x1): normal vector defined by ring atoms


    """
    def __init__(self,name,coor,atom_names,BOND_CUTOFF=1.65,ATOM_LIST=['O','C','N','S','P','X']):
        #self.is_sia = is_sia
        self.name = name
        self.coor = coor
        #print(self.coor)
        self.atom_names = atom_names

        self.BOND_CUTOFF = BOND_CUTOFF
        self.ATOM_LIST = ATOM_LIST

        #initialize empty variables
        self.atom_onehot = []

        self.adj_mat = []
        self.edges = []
        self.ring_atom = []
        self.ring_onehot = np.zeros(len(coor))
        self.normal = []
        self.com = []


        self.atom_name_to_oneHot()
        #print(self.coor)
        #print(atom_names,self.atom_onehot)
        self.calc_adjacency()
        #p#rint(self.coor)
        #print(self.adj_mat,self.edges)
        self.calc_ring()
        #print(self.coor)
        self.calc_adjacency() # deep copy isn't working ?Fix later? idk, probably not worth the effort. not a computational bottleneck
        #print(self.adj_mat,self.edges)
        #print(self.edges)
        #print(self.coor)
        self.normal = self.get_normal()
        #print(self.coor)

        self.ring_com = self.get_ring_com()

        #self.print_variables()

    def deepcopy(self):
        return copy.deepcopy(self.name), copy.deepcopy(self.coor)

    def print_variables(self):
        print("Name:",self.name)
        print("Coor:",self.coor)
        print("Atom_names:",self.atom_names)
        print("adjacency:",self.adj_mat,'\n',self.edges)
        print("Ring:",self.ring_atom,'\t',self.ring_onehot)
        print('Normal:',self.normal)
        print('COM:',self.ring_com)
        return

    def atom_name_to_oneHot(self):
        a = []
        for ii in range(len(self.atom_names)):
            a.append(np.zeros(len(self.ATOM_LIST)));
            for jj in range(len(self.ATOM_LIST)):
                if self.ATOM_LIST[jj] in self.atom_names[ii]:
                    a[ii][jj] = 1;
                    break;
                #if no atom exists in the list put it in the 'x' / other category
                if np.sum(a[ii]) == 0:
                    a[ii][-1] = 1;
        self.atom_onehot = a
        return


    def calc_adjacency(self):
        #get the adjacency matrix and edge list of the carb

        #calculate atom-atom distances and set cutoffs
        dm = distance_matrix(self.coor,self.coor)
        #print(dm)
        adj_mat = dm < self.BOND_CUTOFF;
        #no self interactions
        for i in range(len(adj_mat)):
            adj_mat[i,i] = 0

        #get the list of the adjacency matrix
        edge_list = [];
        for ii in range(len(adj_mat)):
            edge_list.append([])
            for jj in range(len(adj_mat)):
                if adj_mat[ii,jj]:
                    edge_list[ii].append(jj)

        #store local variables into class variables
        self.adj_mat = adj_mat
        self.edges = edge_list
        return


    def get_ring_com(self):
        #print(self.ring_onehot)
        ring = self.coor[self.ring_atom,:]
        self.ring_com = np.mean(ring,axis=0)
        #print(ring)
        return self.ring_com;

    #calculates normal vector of the ring
    #done as first ring atom, 3rd ring atom, last ring atom
    def get_normal(self):
        #print(self.ring_onehot)
        ring = self.coor[self.ring_atom]
        a1 = ring[0,:]
        a2 = ring[2,:]
        a3 = ring[-1,:]
        v1 = a1 - a2
        v2 = a2 - a3
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        normal = np.cross(v1,v2)
        return normal / np.linalg.norm(normal)

    def rotate(self,anchor,R):
        """
        Args:
            anchor (arr 1x3): anchor point of the rotation
            R (3x3 rot mat): rotation matrix
        """
        new_coor = R.apply(self.coor)
        new_anc = R.apply(anchor)
        dx = anchor - new_anc
        #print(dx)
        self.coor = new_coor + dx

        #recalculate ring properties
        self.normal = self.get_normal()
        self.ring_com = self.get_ring_com()
        return

    def translate(self,dx):
        self.coor = self.coor + dx
        self.normal = self.get_normal()
        self.ring_com = self.get_ring_com()
        return;

    #recursive algo to get cycle of the graph
    def visit(self,n,edge_list,visited,st):
        """
        Args:
            n - node we are searching from
            edge_list - adjacency of each node, is periodically
                modified to remove connection to parent coming from
            st - start node
        Returns:
            arr - array of the cycle found
        """
        #print(edge_list)
        #print(n)

        if n == st and visited[st] == True:
            return [n]

        visited[n] = True
        r = False
        arr = []
        #print(n,edge_list[n],visited)

        for e in edge_list[n]:
            #if n in edge_list[e]:
            try:
                edge_list[e].remove(n)
            except:
                continue;
            #print('\t',e)

            r = self.visit(e,edge_list,visited,st)
            #print('\t\t',r)

            if type(r) != bool:
                arr.append(n)
                for j in r:
                    arr.append(j)

        if arr == []:
            return False

        return arr

    def calc_ring(self):
        #gets the ring atoms, calls recursive visit function
        ring = self.visit(0,self.edges.copy(),np.zeros(len(self.coor)),0)
        #print(ring)
        self.ring_atom  = np.unique(ring).astype(int)
        #print(self.ring_atom)
        for ii in self.ring_atom:
            self.ring_onehot[int(ii)] = 1
        self.ring_onehot = self.ring_onehot.astype(int)
        return

#A = poly(monomer_array)
class poly():

    """
    Args:
        monos (arr nx1 mono): monomers of the glycan chain

    Variables:

        adj_mat (arr nxn): adjacency matrix
        edges (arr nx?): connection edges of the shared atoms
        link_atoms (arr nxn): defines atom number of shared "linker" atoms - C or O
        DIST_CUTOFF (float): defines inter-res atom connection distance cutoff for bond
        angle_diff (arr nxn): defines difference between consecutive monomers

    """

    def __init__(self,monos,DIST_CUTOFF=.5):
        self.monos = monos;

        self.adj_mat = np.zeros( (len(monos),len(monos)) );
        self.link_atoms = np.zeros( (len(monos),len(monos)) );
        self.edges = []
        self.DIST_CUTOFF=DIST_CUTOFF
        self.angle_diff = []

        self.calc_adj()
        self.calc_normal_angles()

        #self.print_variables()

    #deep copy of the monomers
    def deepcopy(self):
        #to create a deep copy
        # p1 = poly( p.deepcopy() )
        return copy.deepcopy(self.monos)

    def print_variables(self):
        #print("Name:",self.name)
        print("adjacency:\n",self.adj_mat,'\n',self.edges)
        print("link_atoms:\n",self.link_atoms)
        return

    def calc_adj(self):
        #calculate adjacency between monomers

        #gather all coordinates of the adjacency matrix
        coor = np.zeros(3)
        res = []
        a_num = []
        for i in range(len(self.monos)):
            c = self.monos[i].coor #idc, keeping these variables public. Sue me
            for j in range(len(c)):
                res.append(i)
                a_num.append(j) # atom_number
            if len(coor) < 10:
                coor = c
            else:
                coor = np.row_stack((coor,c))
        #print(res)
        dm = distance_matrix(coor,coor)
        #adj = dm < self.monos[i].BOND_CUTOFF # SUE ME DUDE
        adj = dm < self.DIST_CUTOFF
        #print(adj)
        conn = np.where(adj == True)
        for i in range(len(conn[0])):
            ii = int(conn[0][i])
            jj = int(conn[1][i])

            r1 = res[ii]
            r2 = res[jj]
            #print('a')
            if r1 == r2:
                continue;
            #only oxygens connect
            if 'O' not in self.monos[r1].atom_names[a_num[ii]]:
                continue;
            if 'O' not in self.monos[r2].atom_names[a_num[jj]]:
                continue;

            self.adj_mat[r1,r2] = 1
            self.link_atoms[r1,r2] = a_num[ii]
            self.link_atoms[r2,r1] = a_num[jj]
            #print(a[ii],a[jj],r1,r2)


        for ii in range(len(self.adj_mat)):
            self.edges.append([])
            for jj in range(len(self.adj_mat[ii])):
                if self.adj_mat[ii,jj]:
                    self.edges[ii].append(jj)

        self.link_atoms = self.link_atoms.astype(int)
        return;

    #gets com of rings (1x3)
    def get_com(self):
        com = 0
        for i in self.monos:
            com += i.get_ring_com();
        return com / len(self.monos)

    #gets com of all_atoms (1x3)
    def get_atom_com(self):
        com, ind = 0,0
        for i in self.monos:
            com += np.sum(i.coor,axis=0);
            ind += len(i.coor)
        return com / ind

    #returns all rings com's (nx3)
    def get_ring_coms(self):
        com = []
        for i in self.monos:
            com.append(i.ring_com);
        return np.array(com)

    #gets atomic coordinates - used for alignment and atom losses (n_atoms x 3)
    def get_atom_coor(self):
        coor = np.zeros(3)
        for i in range(len(self.monos)):
            c = self.monos[i].coor #idc, keeping these variables public. Sue me
            if len(coor) < 4:
                coor = c
            else:
                coor = np.row_stack((coor,c))
        return coor


    #recursively find all connections in the polymer chain
    # rotates:  R1 -> O ''->'' R2
    # anchor: O
    def find_all_connections(self,r1,r2,arr=[]):
        #print(r1,r2)
        #arr = []
        arr.append(r2)
        #print(r2,self.edges[r2])
        for e in self.edges[r2]:

            if e != r1 and e not in arr:
                #print('\t',r1,r2,e,'\t',arr)
                arr.append(e)
                a = self.find_all_connections(r2,e,arr)
                for kk in a:
                    if kk not in arr:
                        arr.append(kk)
            #else:
                #print('banned:\t',r1,'\t',r2)
        return np.unique(arr).astype(int)

    #translate the entire carbohydrate
    def translation(self,dx):
        for m in self.monos:
            m.translate(dx)
        return

    #rotate the entire carbohydrate
    def euler_rotation(self,R):
        for m in self.monos:
            m.coor = R.apply(m.coor)
        return


    def calc_normal_angles(self):
        #calculates the difference between the normal vectors of connected monomers
        n = [];
        for i in self.monos:
            n.append(i.get_normal())
        diff = np.zeros((len(self.monos),len(self.monos)))
        for c in range(len(self.edges)):
            for e in self.edges[c]:
                diff[c,e] = np.arccos(n[c].dot(n[e])) #its normalized so fuck it
        self.angle_diff = diff
        return diff;


    #rotate a monomer and all lever arm residues in the chain by some change in phi
    def dihedral_rotation(self,r1,r2,d_phi,degrees=True):
        #find the lever arm to rotate
        monos_to_rot = self.find_all_connections(r1,r2,arr=[])
        anchor1 = self.monos[r1].coor[self.link_atoms[r1,r2],:]
        anchor2 = self.monos[r2].coor[self.link_atoms[r2,r1],:]
        #print(r1,r2,'\tmono_to_rot: ',monos_to_rot)

        #determine which is the oxygen and get the connection needed
        a1_name = self.monos[r1].atom_names[self.link_atoms[r1,r2]]
        a2_name = self.monos[r2].atom_names[self.link_atoms[r2,r1]]
        anchor = []

        #print('r1 atom:',a1_name,'r2 atom:',a2_name)

        #same position!!!
        #print(anchor1,anchor2)
        #print( self.monos[r1].coor[ self.link_atoms[r1,r2],: ], self.monos[r2].coor[ self.link_atoms[r2,r1],: ] )

        #same atom, different name!
        #a1_name = self.monos[r1].atom_names[self.link_atoms[r2,r1]]
        #a2_name = self.monos[r2].atom_names[self.link_atoms[r1,r2]]
        #print(a1_name,a2_name)

        vec = []

        #if 'O' in a1_name:
            #print(r1)
            #print(self.monos[r1].edges[self.link_atoms[r1,r2]])
        #print( self.monos[r1].atom_names[ self.monos[r1].edges[self.link_atoms[r1,r2]][0] ])
        conn = self.monos[r1].coor[ self.monos[r1].edges[self.link_atoms[r1,r2]][0] ,:]
        vec = anchor1 - conn
        #print(conn,vec)
        anchor = anchor1

        #print(self.monos[r1].edges[self.link_atoms[r1,r2]])

        #elif 'O' in a2_name:
        #    conn = self.monos[r2].coor[ self.monos[r2].edges[self.link_atoms[r2,r1]][0] ,:]
        #    vec = anchor2 - conn
        #    anchor = anchor2

        #else:
        #    print('OOF i fucked up')

        #print(rotation_matrix_from_axis([0,0,1],45,degrees=True))

        #get the rotation matrix
        R = rotation_matrix_from_axis(vec, d_phi,degrees=degrees)
        #print(d_phi,vec)
        #print(R)
        #print(R)
        R = Rot.from_matrix(R)
        #print(R)

        #rotate all monomers in the lever arm
        for m in monos_to_rot:
            self.monos[m].rotate(anchor,R);

    def output_pdb(self):

        '''

        #
        #  1 -  6        Record name     "ATOM  "
        #  7 - 11        Integer         Atom serial number.
        # 13 - 16        Atom            Atom name.
        # 17             Character       Alternate location indicator.
        # 18 - 20        Residue name    Residue name.
        # 22             Character       Chain identifier.
        # 23 - 26        Integer         Residue sequence number.
        # 27             AChar           Code for insertion of residues.
        # 31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.
        # 39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.
        # 47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.

        '''
        anum = 1;
        resnum = 1;
        out = ""

        for m in self.monos:

            for a in range(len(m.coor)):
                out += "ATOM  "
                out += str(anum).rjust(5) + ' '
                out += m.atom_names[a].rjust(4)
                out += ' ' # no alt location indicator
                out += m.name.ljust(4)
                out += 'A'
                out += str(resnum).rjust(4)
                out += '   '
                #print(m.coor[a,:])
                c = str( round(m.coor[a,0],3) )
                while len( c.split('.')[1] ) < 3:
                    c += '0'
                out += c.rjust(8)
                c = str( round(m.coor[a,1],3) )
                while len( c.split('.')[1] ) < 3:
                    c += '0'
                out += c.rjust(8)
                c = str( round(m.coor[a,2],3) )
                while len( c.split('.')[1] ) < 3:
                    c += '0'
                out += c.rjust(8)


                #out += str( round(m.coor[a,1],3) ).rjust(8)
                #out += str( round(m.coor[a,2],3) ).rjust(8)

                out += '\n'
                anum += 1
            resnum += 1

        return out

def align(a,b):
    '''
    Aligns polymer A to polymer B
    e.g. A is translated and rotated
    Kabsh-Umeyama Algortihm

        Args:
            a (polymer)
            b (polymer)
        Return:
            null / updated B
    '''
    #cetner both on origin
    com_a = a.get_atom_com()
    com_b = b.get_atom_com()
    #print(com_a)
    a.translation(0-com_a)
    b.translation(0-com_b)

    #print(com_a,a.get_atom_com())


    #dx = np.array(com_b - com_a);

    #print(com_a,com_b,dx)
    #a.translation(dx)

    #com_a = a.get_com()
    #com_b = b.get_com()
    #print(com_a,com_b,dx)

    a_coor = a.get_atom_coor()
    b_coor = b.get_atom_coor()

    #a_sum = np.sum(a_coor,axis=0)
    #b_sum = np.sum(b_coor,axis=0)

    H = np.dot(np.transpose(b_coor),a_coor) / len(a_coor)

    #print(a_coor,b_coor)
    #print(a.get_ring_coms(),b.get_ring_coms())

    U, S, V = np.linalg.svd(H)
    #print(np.shape(U),np.shape(V),np.shape(S))
    R = np.dot(U,V)

    #remove reflection
    if np.linalg.det(R) < 0:
        F = np.eye(3)
        F[2,2] = -1
        R1 = np.dot(U,F)
        R = np.dot(R1,V)

    #print(R)
    R = Rot.from_matrix(R)
    a.euler_rotation(R)

    com_a = a.get_atom_com()
    com_b = b.get_atom_com()

    dx = np.array(com_b - com_a);
    #print(com_a,com_b,dx)
    a.translation(dx)
    #print(a.get_atom_com())

    return;

def atom_rmsd(a,b):
    '''
    Calculates RMSD on atom-atom basis

        Args:
            a (polymer)
            b (polymer)
        Return:
            rmsd (float)
    '''

    a_coor = a.get_atom_coor()
    b_coor = b.get_atom_coor()

    dx = a_coor - b_coor;
    #print(dx[:10,:])

    dx = np.linalg.norm(dx,axis=0);

    rmsd = np.sqrt( np.sum( dx**2 ) / len(dx) )
    return rmsd

def ring_rmsd(a,b):
    '''
    Calculates RMSD on ring-ring basis

        Args:
            a (polymer)
            b (polymer)
        Return:
            rmsd (float)
    '''

    a_coor = a.get_ring_coms()
    b_coor = b.get_ring_coms()

    dx = a_coor - b_coor;
    dx = np.linalg.norm(dx,axis=0);
    rmsd = np.sqrt( np.sum( dx**2 ) / len(dx) )
    return rmsd

#calculates the angle difference between the angle differences of polys
def get_angle_diff(a,b):
    na = a.calc_normal_angles()
    nb = b.calc_normal_angles()

    #ind = np.where(na != 0);
    #print(ind)
    #print(na[ind])
    #print(nb[ind])

    #print( abs(na - nb) )
    #print( np.sin(abs(na - nb))**2 )

    return( np.sin(abs(na - nb))**2 )



def rotation_matrix_from_axis(axis, theta, degrees=False):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if degrees:
        theta *= np.pi / 180

    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def pyrosetta_to_poly(pose):
    polymer = []
    for ii in range(1,pose.size()+1):
        r = pose.residue(ii)
        n = []
        c = []

        for a in range(1,r.natoms()+1):
            name = r.atom_name(a)

            #only the real heavy atoms
            if "H" in name or "V" in name:
                continue;

            #print(a,r.atom_name(a),r.xyz(a))

            n.append(name)
            c.append(np.array(r.xyz(a)))

        c = np.array(c)
        #print(r.name(),n,c)
        #print(r)
        #print(r.name3())
        m = mono(r.name3(),c,n)
        #print(m.ring_atom)
        polymer.append(m)

    return poly(polymer)

if __name__ == '__main__':

    #simple test of monos and polys
    coor = np.array( [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                     [ 1.53000000e+00,  0.00000000e+00,  0.00000000e+00],
                     [ 1.64690405e+00,  1.48540683e+00,  0.00000000e+00],
                     [ 2.98008524e+00,  1.94445793e+00,  1.58353531e-16],
                     [ 8.33917003e-01,  1.71191041e+00, -1.30009545e+00],
                     [-3.68861556e-01,  9.34799852e-01, -1.07222943e+00],
                     [-6.33600895e-01,  6.35403397e-01,  1.10055097e+00],
                     [ 2.18411390e+00, -6.34911806e-01,  1.10178129e+00],
                     [ 6.34378179e-01,  3.17849385e+00, -1.79296185e+00],
                     [-5.49125617e-02,  3.99521621e+00, -8.27848979e-01],
                     [-1.83853949e-01,  3.40645874e+00, -3.10144501e+00],
                     [-2.94593960e-01,  4.80014399e+00, -3.44637701e+00]] )

    names = ['C','O','Q','N','N','N','N','N','N','N']
    m = mono('bruh',coor,names)
    p = poly([m])
