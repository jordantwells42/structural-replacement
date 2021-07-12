from pyrosetta import *

from pyrosetta.rosetta.numeric import xyzMatrix_double_t, xyzVector_double_t
import math


class CollisionGrid():
    
    VDW_R = {
        "H": 1.2, 
        "C": 1.7,   
        "N": 1.55,   
        "O": 1.52,   
        "F": 1.47,   
        "P": 1.8,    
        "S": 1.8,   
        "CL": 1.75,   
        "CU": 1.4,
        "BR": 1.7,
        "I": 1.7
    }
    
    def __init__(self, pose, bin_width = 1, vdw_modifier = 0.7, include_sc = False):
        """
        pose: Pose to calculate a grid for
        bin_width: size of the grid boxes
        vdw_mofifier: by what factor to reduce vdw radii
        include_sc: boolean to determine whether to check just backbone or all atoms


        """
        self.pose = pose
        self.bin_width = bin_width
        self.vdw_modifier = vdw_modifier
        self.include_sc = include_sc

        self.grid = self.precalculate_grid()


    def precalculate_grid(self):
        """
        Precalculates a hash map grid of the pose, voxelizing the atoms based on
        a reduced VDW radius (*note a grid is considered to be "filled" if its
        center is contained within the sphere)

        """
        grid = {}
        for resid in range(1, self.pose.size() + 1):

            # Only backbone atoms (first four atoms in residue object)
            if self.include_sc:
                natoms = self.pose.residue(resid).natoms()
            else:
                natoms = 4

            for i in range(1, natoms + 1):
                aid = AtomID(i, resid)
                atom_coords = (self.pose.xyz(aid))
                elem = self.pose.residue(resid).atom_type(i).element()
                if elem not in self.VDW_R:
                    R = 2
                else:
                    R = self.VDW_R[elem]
                    R *= self.vdw_modifier

                x_extremes = [atom_coords.x - R, atom_coords.x + R]
                y_extremes = [atom_coords.y - R, atom_coords.y + R]
                z_extremes = [atom_coords.z - R, atom_coords.z + R]

                # Only grab feasible grid coordinates to cut down intersection checks later         
                xs = CollisionGrid._multiples_about_range(self.bin_width, x_extremes)
                ys = CollisionGrid._multiples_about_range(self.bin_width, y_extremes)
                zs = CollisionGrid._multiples_about_range(self.bin_width, z_extremes)
            
                for box in self._grids_whose_center_intersects(R, atom_coords, [f"{x}_{y}_{z}" for x in xs for y in ys for z in zs]):
                    if box not in grid:
                        grid[box] = resid
                
        return grid


    def check_collision_matrix(self, matrix):
        """
        Checks the collision of a residue object (presumably a ligand) against a precalculated grid
        (*note a grid is considered to be "filled" if its center is contained within the sphere)

        Arguments:
        matrix: A matrix of coordinates [[x1,y1,z1],[x2,y2,z2],..]

        Returns:
        boolean, True meaning it collides False meaning it does not
        """
        for vector in matrix:            
            x,y,z = vector
            
            R = 1.5
            R *= self.vdw_modifier
            
            x_extremes = [x - R, x + R]
            y_extremes = [y - R, y + R]
            z_extremes = [z - R, z + R]

            # Only grab feasible grid coordinates to cut down intersection checks later   
            xs = CollisionGrid._multiples_about_range(self.bin_width, x_extremes)
            ys = CollisionGrid._multiples_about_range(self.bin_width, y_extremes)
            zs = CollisionGrid._multiples_about_range(self.bin_width, z_extremes)

            for box in self._grids_whose_center_intersects(R, vector, [f"{x}_{y}_{z}" for x in xs for y in ys for z in zs]):
                if box in self.grid:
                    return True
            
        return False


    def check_collision(self, res):
        """
        Checks the collision of a residue object (presumably a ligand) against a precalculated grid
        (*note a grid is considered to be "filled" if its center is contained within the sphere)

        Arguments:
        res: Residue object whose atoms will be checked against the grid

        Returns:
        boolean, True meaning it collides False meaning it does not
        """

        for i in range(1, res.natoms() + 1):
            atom_coords = (res.atom(i).xyz())
            elem = res.atom_type(i).element()
            
            R = self.VDW_R[elem]
            R *= self.vdw_modifier

            x_extremes = [atom_coords.x - R, atom_coords.x + R]
            y_extremes = [atom_coords.y - R, atom_coords.y + R]
            z_extremes = [atom_coords.z - R, atom_coords.z + R]

            # Only grab feasible grid coordinates to cut down intersection checks later   
            xs = CollisionGrid._multiples_about_range(self.bin_width, x_extremes)
            ys = CollisionGrid._multiples_about_range(self.bin_width, y_extremes)
            zs = CollisionGrid._multiples_about_range(self.bin_width, z_extremes)

            #print(xs, ys, zs)
            for box in self._grids_whose_center_intersects(R, atom_coords, [f"{x}_{y}_{z}" for x in xs for y in ys for z in zs]):
                if box in self.grid:
                    return True
        
        return False


    def score_collision(self, res):
        """
        Checks the collision of a residue object (presumably a ligand) against a precalculated grid
        (*note a grid is considered to be "filled" if its center is contained within the sphere)

        Arguments:
        res: Residue object whose atoms will be checked against the grid

        Returns:
        int, number of collisions with backbone
        """

        score = 0

        for i in range(1, res.natoms() + 1):
            atom_coords = (res.atom(i).xyz())
            elem = res.atom_type(i).element()
            
            R = self.VDW_R[elem]
            R *= self.vdw_modifier

            x_extremes = [atom_coords.x - R, atom_coords.x + R]
            y_extremes = [atom_coords.y - R, atom_coords.y + R]
            z_extremes = [atom_coords.z - R, atom_coords.z + R]


            xs = CollisionGrid._multiples_about_range(self.bin_width, x_extremes)
            ys = CollisionGrid._multiples_about_range(self.bin_width, y_extremes)
            zs = CollisionGrid._multiples_about_range(self.bin_width, z_extremes)

            
            for box in self._grids_whose_center_intersects(R, atom_coords, [f"{x}_{y}_{z}" for x in xs for y in ys for z in zs]):
                if box in self.grid:
                    score += 1
        
        return score


    def _multiples_about_range(base, range):
        num = int(range[0]/ base)*base
        nums = []
        
        while num <= range[1]:
            nums.append(num)
            num += base
            
        return nums + [num]


    def _grids_whose_center_intersects(self, R, center, grid_boxes):
        if not grid_boxes:
            return []
        intersects = []
        c_x, c_y, c_z = center
        for box in grid_boxes:
            x,y,z = [float(e) + self.bin_width/2 for e in box.split("_") if e]
            
            #print(x, y, z)
            #print(center.x, center.y, center.z)
            dist_2 = (x - c_x)**2 + (y - c_y)**2 + (z - c_z)**2
            if dist_2 <= R**2:
                intersects.append(box)

        return intersects


def main():
    pass
    
if __name__ == "__main__":
    init()
    main()