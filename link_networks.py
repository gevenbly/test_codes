
from numpy import linalg as LA
import tensornetwork as tn
from helper_functs import trunct_eigh, orthog


#######################################
def LiftHam(ham, u, w):

  tensors = [ham,u,u,u.conj(),u.conj(),w,w,w,w.conj(),w.conj(),w.conj()] 
  
  # Network 1 - leading cost: (chi^4)*(chi_p^5) 
  connects1 = [[5,6,7,2,3,4],[1,2,9,10],[3,4,11,12],[1,5,13,14],[6,7,15,16],
              [10,11,-5],[8,9,-4],[12,17,-6],[14,15,-2],[8,13,-1],[16,17,-3]] 
  con_order1 = [10,6,7,3,4,2,11,17,8,5,1,15,14,16,12,9,13]
  
  # Network 2 - leading cost: (chi^4)*(chi_p^5) 
  connects2 = [[4,5,6,1,2,3],[1,2,8,9],[3,17,10,11],[4,5,12,13],[6,17,14,15],
              [9,10,-5],[7,8,-4],[11,16,-6],[13,14,-2],[7,12,-1],[15,16,-3]] 
  con_order2 = [14,4,5,1,2,16,6,13,7,3,17,9,10,15,11,12,8]
  
  return (tn.ncon(tensors,connects1,con_order1) +
          tn.ncon(tensors,connects2,con_order2))
  

#######################################
def LowerDensity(u, w, rho1):

  tensors = [u,u,u.conj(),u.conj(),w,w,w,w.conj(),w.conj(),w.conj(),rho1]
  
  # Network 3 - leading cost: (chi^4)*(chi_p^5) 
  connects1 = [[4,-4,9,10],[-5,-6,11,12],[4,-1,13,14],[-2,-3,15,16],[10,11,2],
              [8,9,1],[12,17,3],[14,15,6],[8,13,5],[16,17,7],[5,6,7,1,2,3]] 
  con_order1 = [8,1,5,17,14,7,3,2,9,10,4,13,6,16,15,11,12]
  
  # Network 4 - leading cost: (chi^4)*(chi_p^5) 
  connects2 = [[-4,-5,8,9],[-6,17,10,11],[-1,-2,12,13],[-3,17,14,15],[9,10,2],
              [7,8,1],[11,16,3],[13,14,5],[7,12,4],[15,16,6],[4,5,6,1,2,3]] 
  con_order2 = [10,16,7,1,4,6,3,5,14,15,17,11,2,12,13,8,9]
 
  return 0.5*(tn.ncon(tensors,connects1,con_order1) + 
              tn.ncon(tensors,connects2,con_order2))

#######################################
def RightLink(ham, u, w, rho1, chi_m):

  # Network 1 - leading cost: (chi^4)*(chi_p^2) 
  tensors = [w, rho1, w.conj()] 
  connects = [[4,-3,1],[3,2,-2,3,1,-4],[4,-1,2]] 
  con_order = [3,2,4,1] 
  rhotemp = tn.ncon(tensors,connects,con_order) 
  _, proj = trunct_eigh(rhotemp, chi_m)

  tensors = [ham,u,u.conj(),u.conj(),w,w,w.conj(),w.conj(),w.conj(),rho1,proj]
  
  # Network 1 - leading cost: (chi^5)*(chi_p^2)*(chi_c^1) 
  connects = [[5,6,7,4,-1,-2],[3,4,12,13],[3,5,14,15],[6,7,16,17],[13,19,2],
              [11,12,1],[15,16,9],[11,14,8],[17,-3,10],[8,9,10,1,2,18],[19,18,-4]]
  cont_order = [11,1,8,19,2,18,15,12,13,3,14,9,6,7,17,5,4,16,10]
  temp1 = tn.ncon(tensors,connects,cont_order)

  # Network 2 - leading cost: (chi^3)*(chi_p^5)*(chi_c^1) 
  connects = [[5,6,7,3,4,-1],[3,4,12,13],[5,6,14,15],[7,-2,16,17],[13,18,2],
              [11,12,1],[15,16,9],[11,14,8],[17,-3,10],[8,9,10,1,2,19],[18,19,-4]]
  cont_order = [11,1,8,18,2,19,5,6,3,4,9,14,15,12,13,17,7,10,16]
  temp2 = tn.ncon(tensors,connects,cont_order)

  # Network 3 - leading cost: (chi^5)*(chi_p^2)*(chi_c^1) 
  connects = [[5,6,7,-2,3,4],[3,4,-3,12],[-1,5,13,14],[6,7,15,16],[11,18,1],
              [12,17,2],[14,15,9],[11,13,8],[16,17,10],[8,9,10,1,19,2],[18,19,-4]]
  cont_order = [11,1,8,18,19,17,10,2,9,13,14,16,15,3,4,5,6,7,12]
  temp3 = tn.ncon(tensors,connects,cont_order)

  # Network 4 - leading cost: (chi^6)*(chi_p^1)*(chi_c^1) 
  connects = [[4,5,6,-1,-2,3],[3,17,-3,11],[4,5,12,13],[6,17,14,15],[10,18,1],
              [11,16,2],[13,14,8],[10,12,7],[15,16,9],[7,8,9,1,19,2],[18,19,-4]]
  cont_order = [10,1,7,18,19,16,9,2,8,14,15,17,11,12,13,4,5,6,3]
  temp4 = tn.ncon(tensors,connects,cont_order)
  
  gam1 = tn.ncon([temp1+temp2+temp3+temp4,proj],[[-1,-2,-3,1],[-4,-5,1]])
  
  tensors = [ham,u,u,u.conj(),u.conj(),w,w,w.conj(),w.conj(),w.conj(),rho1]

  # Network 2 - leading cost: (chi^4)*(chi_p^5) 
  connects = [[7,8,9,4,5,6],[3,4,-2,13],[5,6,14,15],[3,7,16,17],[8,9,18,19],
              [13,14,1],[15,20,2],[17,18,11],[-1,16,10],[19,20,12],[10,11,12,-3,1,2]]
  cont_order = [20,8,9,13,5,6,4,14,7,3,18,17,19,15,1,11,2,12,16,10]
  temp1 = tn.ncon(tensors,connects,cont_order)

  # Network 3 - leading cost: (chi^4)*(chi_p^5) 
  connects = [[6,7,8,3,4,5],[3,4,-2,12],[5,20,13,14],[6,7,15,16],[8,20,17,18],
              [12,13,1],[14,19,2],[16,17,10],[-1,15,9],[18,19,11],[9,10,11,-3,1,2]]
  cont_order = [6,7,3,4,17,8,16,5,20,19,12,13,18,14,10,1,2,11,15,9]
  temp2 = tn.ncon(tensors,connects,cont_order)
  
  gam2 = temp1 + temp2
  
  return gam1, gam2

#######################################
def LeftLink(ham, u, w, rho1, chi_m):

  # Network 1 - leading cost: (chi^5)*(chi_p^1) 
  tensors = [w,w.conj(),rho1]
  connects = [[-4,3,1],[-2,3,2],[-1,2,4,-3,1,4]]
  con_order = [4,1,3,2]
  rhotemp = tn.ncon(tensors,connects,con_order)
  _, proj = trunct_eigh(rhotemp, chi_m)
  
  tensors = [ham,u,u.conj(),u.conj(),w,w,w.conj(),w.conj(),w.conj(),rho1,proj]
  
  # Network 1 - leading cost: (chi^3)*(chi_p^5)*(chi_c^1) 
  connects = [[5,6,7,-3,3,4],[3,4,11,12],[-2,5,13,14],[6,7,15,16],[19,11,1],[12,17,2],[14,15,9],[-1,13,8],[16,17,10],[8,9,10,18,1,2],[18,19,-4]] 
  cont_order = [17,19,2,10,1,18,3,4,13,9,6,7,11,12,15,16,5,8,14]
  temp1 = tn.ncon(tensors,connects,cont_order)
  
  # Network 2 - leading cost: (chi^5)*(chi_p^2)*(chi_c^1) 
  connects = [[4,5,6,-2,-3,3],[3,17,10,11],[4,5,12,13],[6,17,14,15],[18,10,1],[11,16,2],[13,14,8],[-1,12,7],[15,16,9],[7,8,9,19,1,2],[19,18,-4]]
  cont_order = [16,2,9,4,5,18,1,19,10,11,14,17,15,8,12,6,3,13,7]
  temp2 = tn.ncon(tensors,connects,cont_order)
  
  # Network 3 - leading cost: (chi^6)*(chi_p^1)*(chi_c^1) 
  connects = [[5,6,7,4,-2,-3],[3,4,12,-1],[3,5,13,14],[6,7,15,16],[11,12,1],[18,17,2],[14,15,9],[11,13,8],[16,17,10],[8,9,10,1,19,2],[19,18,-4]]
  cont_order = [17,2,10,18,19,11,1,8,9,13,14,3,12,16,15,5,6,7,4]
  temp3 = tn.ncon(tensors,connects,cont_order)
  
  # Network 4 - leading cost: (chi^5)*(chi_p^2)*(chi_c^1) 
  connects = [[5,6,7,3,4,-2],[3,4,12,-1],[5,6,13,14],[7,-3,15,16],[11,12,1],[18,17,2],[14,15,9],[11,13,8],[16,17,10],[8,9,10,1,19,2],[19,18,-4]]
  cont_order = [17,2,10,11,18,19,1,8,9,3,4,15,16,13,14,5,6,7,12]
  temp4 = tn.ncon(tensors,connects,cont_order)
  
  gam1 = tn.ncon([temp1+temp2+temp3+temp4,proj],[[-1,-2,-3,1],[-4,-5,1]])
  
  tensors = [ham,u,u,u.conj(),u.conj(),w,w,w.conj(),w.conj(),w.conj(),rho1]
  
  # Network 2 - leading cost: (chi^4)*(chi_p^5) 
  connects = [[7,8,9,4,5,6],[3,4,14,15],[5,6,16,-1],[3,7,17,18],[8,9,19,20],[15,16,2],[13,14,1],[18,19,11],[13,17,10],[20,-2,12],[10,11,12,1,2,-3]]
  cont_order = [5,6,8,9,15,4,16,7,3,13,19,18,14,17,2,11,1,10,20,12]
  temp1 = tn.ncon(tensors,connects,cont_order)
  
  # Network 3 - leading cost: (chi^4)*(chi_p^5) 
  connects = [[6,7,8,3,4,5],[3,4,13,14],[5,20,15,-1],[6,7,16,17],[8,20,18,19],[14,15,2],[12,13,1],[17,18,10],[12,16,9],[19,-2,11],[9,10,11,1,2,-3]] 
  cont_order = [15,3,4,6,7,5,14,12,8,20,17,18,1,9,13,16,2,10,19,11]
  temp2 = tn.ncon(tensors,connects,cont_order)
  
  gam2 = temp1 + temp2
  
  return gam1, gam2

def CenterLink(ham, u, w, u1, w1, rho2, chi, chi_m):

  # (chi^4)*(chi_p^5)
  tensors = [ham,u,u,u.conj(),u.conj(),w.conj()] 
  connects = [[5,6,7,2,3,4],[1,2,-4,-5],[3,4,-6,-7],[1,5,-1,8],[6,7,9,-3],[8,9,-2]] 
  con_order = [8,3,4,6,7,5,9,2,1] 
  temp1 = tn.ncon(tensors,connects,con_order) 
  
  # (chi^4)*(chi_p^5)
  tensors = [ham,u,u,u.conj(),u.conj(),w.conj()] 
  connects = [[4,5,6,1,2,3],[1,2,-4,-5],[3,9,-6,-7],[4,5,-1,7],[6,9,8,-3],[7,8,-2]] 
  con_order = [4,5,1,2,8,6,7,3,9] 
  ham7 = temp1 + tn.ncon(tensors,connects,con_order) 
  
  # (chi^4)*(chi_p^2) 
  tensors = [w1,w1,rho2,w1.conj(),w1.conj()] 
  connects = [[-4,6,2],[7,-3,1],[3,4,5,1,2,5],[-2,6,4],[7,-1,3]] 
  cont_order = [5,7,6,2,4,3,1] 
  rhotemp = tn.ncon(tensors,connects,cont_order) 
  _, proj = trunct_eigh(rhotemp, chi_m)
  
  # Network 1 - leading cost: (chi^2)*(chi_p^6)*(chi_c^1) 
  tensors = [w,w.conj(),w.conj(),u1,u1.conj(),u1.conj(),w1,w1,w1,w1.conj(),w1.conj(),w1.conj(),rho2,ham7,proj] 
  connects = [[18,1,2],[-1,19,4],[20,1,3],[2,25,5,6],[4,21,7,8],[3,25,9,22],[24,5,13],[11,23,12],[6,10,14],[8,9,16],[11,7,15],[22,10,17],[15,16,17,12,13,14],[19,21,20,-2,-3,-4,18],[23,24,-5]] 
  cont_order = [9,11,1,12,15,24,10,17,14,13,23,5,6,25,16,22,2,3,7,8,18,20,21,4,19] 
  temp1 = tn.ncon(tensors,connects,cont_order) 
  
  # Network 2 - leading cost: (chi^3)*(chi_p^6) 
  tensors = [w,w,w.conj(),w.conj(),u1,u1.conj(),u1.conj(),w1,w1,w1,w1.conj(),w1.conj(),w1.conj(),rho2,ham7,w.conj(),proj] 
  connects = [[20,21,3],[1,19,2],[1,22,5],[23,-2,4],[2,3,6,7],[5,24,8,9],[4,25,10,26],[7,27,14],[12,6,13],[28,11,15],[9,10,17],[12,8,16],[26,11,18],[16,17,18,13,14,15],[22,24,23,19,20,21,-1],[-3,-4,25],[27,28,-5]] 
  cont_order = [5,4,1,12,20,21,25,27,11,15,18,13,16,14,28,17,22,24,19,3,2,8,9,6,7,23,26,10] 
  temp2 = tn.ncon(tensors,connects,cont_order) 
  
  # Network 3 - leading cost: (chi^3)*(chi_p^6) 
  tensors = [w,w.conj(),u1,u1.conj(),u1.conj(),w1,w1,w1,w1.conj(),w1.conj(),w1.conj(),rho2,w,w.conj(),w.conj(),ham7,proj] 
  connects = [[21,22,1],[-1,-2,2],[1,16,3,4],[2,20,5,6],[26,17,7,19],[28,3,11],[9,27,10],[4,8,12],[6,7,14],[9,5,13],[19,8,15],[13,14,15,10,11,12],[23,18,16],[24,18,17],[-3,25,20],[25,26,24,-4,21,22,23],[27,28,-5]] 
  cont_order = [9,20,16,8,21,22,2,28,18,12,15,10,13,11,27,1,24,23,14,26,17,3,4,7,19,25,5,6] 
  temp3 = tn.ncon(tensors,connects,cont_order) 
  
  # Network 4 - leading cost: (chi^2)*(chi_p^6)*(chi_c^1) 
  tensors = [w,u1,u1.conj(),u1.conj(),w1,w1,w1,w1.conj(),w1.conj(),w1.conj(),rho2,w.conj(),w.conj(),ham7,proj] 
  connects = [[18,19,1],[25,1,2,3],[25,17,4,5],[22,15,6,16],[3,23,10],[8,2,9],[24,7,11],[5,6,13],[8,4,12],[16,7,14],[12,13,14,9,10,11],[20,-4,15],[18,21,17],[21,22,20,19,-1,-2,-3],[23,24,-5]] 
  cont_order = [8,18,7,11,14,5,23,9,12,10,24,2,3,25,4,13,1,17,16,6,19,21,22,15,20] 
  temp4 = tn.ncon(tensors,connects,cont_order)
  
  q = orthog(tn.ncon([temp1+temp2+temp3+temp4,proj],[[-1,-2,-3,-4,1],[-5,-6,1]]),4)
  
  # Network 2 - leading cost: (chi_p^8) 
  tensors = [w1,w1,rho2,w1.conj(),w1.conj(),q,q.conj()] 
  connects = [[11,6,2],[7,10,1],[3,4,5,1,2,5],[9,6,4],[7,8,3],[-3,-4,13,12,10,11],[-1,-2,13,12,8,9]] 
  cont_order = [5,6,2,4,7,3,1,9,8,11,10,13,12] 
  temp1 = tn.ncon(tensors,connects,cont_order) 
  
  # Network 3 - leading cost: (chi_p^8) 
  tensors = [w1,w1,rho2,w1.conj(),w1.conj(),q,q.conj()] 
  connects = [[11,6,2],[7,10,1],[3,4,5,1,2,5],[9,6,4],[7,8,3],[13,12,-3,-4,10,11],[13,12,-1,-2,8,9]] 
  cont_order = [5,7,1,3,6,2,4,11,10,9,8,13,12] 
  temp2 = tn.ncon(tensors,connects,cont_order) 
  
  rhotemp = 0.5*(temp1+temp2)
  dtemp, w = trunct_eigh(rhotemp, chi)
  
  u1 = orthog(tn.ncon([q,w.conj(),w.conj()],[[1,2,3,4,-3,-4],[1,2,-1],[3,4,-2]]),2)

  return w, u1

def TopLink(ham,u,w,v,chi):

  # (chi^4)*(chi_p^5)
  tensors = [ham,u,u,u.conj(),u.conj(),w.conj()] 
  connects = [[5,6,7,2,3,4],[1,2,-4,-5],[3,4,-6,-7],[1,5,-1,8],[6,7,9,-3],[8,9,-2]] 
  con_order = [8,3,4,6,7,5,9,2,1] 
  temp1 = tn.ncon(tensors,connects,con_order) 
  
  # (chi^4)*(chi_p^5)
  tensors = [ham,u,u,u.conj(),u.conj(),w.conj()] 
  connects = [[4,5,6,1,2,3],[1,2,-4,-5],[3,9,-6,-7],[4,5,-1,7],[6,9,8,-3],[7,8,-2]] 
  con_order = [4,5,1,2,8,6,7,3,9] 
  ham7 = temp1 + tn.ncon(tensors,connects,con_order) 
  
  tensors = [w,w,w.conj(),w.conj(),ham7,v] 
  connects = [[3,4,-3],[5,1,-4],[-1,2,8],[6,1,9],[2,7,6,-2,3,4,5],[8,7,9]] 
  cont_order = [3,4,9,8,1,2,7,6,5] 
  temp1 = tn.ncon(tensors,connects,cont_order) 
  
  # Network 2 - leading cost: (chi^1)*(chi_p^8) 
  tensors = [w,w,w.conj(),w.conj(),ham7,v] 
  connects = [[1,4,-4],[5,2,-3],[1,3,8],[6,2,9],[3,7,6,4,-1,-2,5],[8,7,9]] 
  cont_order = [9,8,3,6,7,1,4,2,5] 
  temp2 = tn.ncon(tensors,connects,cont_order) 
  
  # Network 3 - leading cost: (chi^3)*(chi_p^5) 
  tensors = [w,w,w.conj(),w.conj(),ham7,v] 
  connects = [[4,5,-4],[1,3,-3],[1,2,8],[6,-2,9],[2,7,6,3,4,5,-1],[8,7,9]] 
  cont_order = [9,8,1,4,5,2,7,6,3] 
  temp3 = tn.ncon(tensors,connects,cont_order)
  
  phitemp = temp1+temp2+temp3
  phitemp = phitemp / LA.norm(phitemp)
  rhotemp = tn.ncon([phitemp,phitemp],[[-1,-2,1,2],[-3,-4,1,2]])
  _, w = trunct_eigh(rhotemp, chi)
  
  return w
