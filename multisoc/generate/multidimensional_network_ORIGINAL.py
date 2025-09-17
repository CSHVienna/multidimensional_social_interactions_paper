import numpy as np
from statsmodels.stats.proportion import proportion_confint
import networkx as nx
from scipy import optimize
import copy
from mpmath import mp
from tqdm import tqdm

##############################################################################
##############################################################################
## THEORY
##############################################################################
##############################################################################

##############################################################################
## Composite homophily
##############################################################################

def composite_H(
    h_mtrx_lst,
    kind,
    p_d = None,
    alpha = None,
    ):

    ## Convert elements of h matrix to np.array just in case
    h_mtrx_lst = [np.array(h) for h in h_mtrx_lst]

    if kind == "one":
        assert np.abs(sum(p_d)-1.0) < 1e-10 ## Verify normalization
    
    if kind == "max":
        assert 0.0 <= alpha <= 1.0
    
    g_vec = [len(h) for h in h_mtrx_lst]

    G = 1
    for g in g_vec:
        G *= g

    H_mtrx = np.zeros((G, G)) + np.nan

    comp_indices = make_composite_index(g_vec)
    assert len(comp_indices[0]) == len(h_mtrx_lst)

    for i_vec in comp_indices:
        for j_vec in comp_indices:
            
            I = comp_index_to_integer(i_vec,g_vec)
            J = comp_index_to_integer(j_vec,g_vec)
            
            if kind == "any":
                H_mtrx[I,J] = composite_H_ij_any(i_vec,j_vec,h_mtrx_lst)
            elif kind == "all":
                H_mtrx[I,J] = composite_H_ij_all(i_vec,j_vec,h_mtrx_lst)
            elif kind == "one":
                H_mtrx[I,J] = composite_H_ij_one(i_vec,j_vec,h_mtrx_lst,p_d)
            elif kind == "max":
                H_mtrx[I,J] = composite_H_ij_max(i_vec,j_vec,h_mtrx_lst,alpha)
            elif kind == "min":
                H_mtrx[I,J] = composite_H_ij_min(i_vec,j_vec,h_mtrx_lst,alpha)
            elif kind == "hierarchy":
                H_mtrx[I,J] = composite_H_ij_hierarchy(i_vec,j_vec,h_mtrx_lst)
            else:
                raise ValueError(f"Interaction kind {kind} invalid.")
    assert not np.any(np.isnan(H_mtrx))
    return H_mtrx

def composite_H_ij_any(i_vec,j_vec,h_mtrx_lst):
    
    Hij = 1.0
    for d, h in enumerate(h_mtrx_lst):
        Hij *= (1.0-h[i_vec[d],j_vec[d]])
    
    return 1.0 - Hij

def composite_H_ij_all(i_vec,j_vec,h_mtrx_lst):
    
    Hij = 1.0
    for d, h in enumerate(h_mtrx_lst):
        Hij *= (h[i_vec[d],j_vec[d]])
    
    return Hij

def composite_H_ij_one(i_vec,j_vec,h_mtrx_lst,p_d):
    
    Hij = 0.0
    for d, h in enumerate(h_mtrx_lst):
        Hij += p_d[d]*h[i_vec[d],j_vec[d]]
    
    return Hij

def composite_H_ij_max(i_vec,j_vec,h_mtrx_lst,alpha):

    D = 1.0*len(h_mtrx_lst)
    h_max = max([h[i_vec[d],j_vec[d]] for d,h in enumerate(h_mtrx_lst)])
    Hij = alpha*h_max

    for d, h in enumerate(h_mtrx_lst):
        Hij += (1.0-alpha) * h[i_vec[d],j_vec[d]] / D
    
    return Hij

def composite_H_ij_min(i_vec,j_vec,h_mtrx_lst,alpha):

    D = 1.0*len(h_mtrx_lst)
    h_min = min([h[i_vec[d],j_vec[d]] for d,h in enumerate(h_mtrx_lst)])
    Hij = alpha*h_min

    for d, h in enumerate(h_mtrx_lst):
        Hij += (1.0-alpha) * h[i_vec[d],j_vec[d]] / D
    
    return Hij

def composite_H_ij_hierarchy(i_vec,j_vec,h_mtrx_lst):

    for d, h in enumerate(h_mtrx_lst):
        if i_vec[d] != j_vec[d]:
            indx = list(i_vec[:d])
            indx.extend([i_vec[d],j_vec[d]])
            indx = tuple(indx)
            Hij = h[indx]
            break ## Only the first dimension with different group membership is considered
        elif d == len(h_mtrx_lst) -1:
            indx = list(i_vec[:d])
            indx.extend([i_vec[d],j_vec[d]])
            indx = tuple(indx)
            Hij = h[indx]
            break

    return Hij

##############################################################################
## Simple homophily and population fraction to link fraction
##############################################################################

def h_f_to_m(h_mtrx,pop_fracs):
    assert len(pop_fracs) == len(h_mtrx)
    m_mtrx = np.zeros_like(h_mtrx)
    n,m = h_mtrx.shape
    for i in range(n):
        norm = (1.0*np.dot(h_mtrx[i,:],pop_fracs))
        for j in range(m):
            m_mtrx[i,j] = pop_fracs[i] * pop_fracs[j] * h_mtrx[i,j] / norm
    return m_mtrx

def all_h_f_to_m(h_mtrx_lst,pop_fracs_lst):
    m_mtrx_lst = []
    for d, h_mtrx in enumerate(h_mtrx_lst):
        pop_fracs = pop_fracs_lst[d]
        m_mtrx_lst.append(h_f_to_m(h_mtrx,pop_fracs))
    return m_mtrx_lst

##############################################################################
## Composite homophily and population fraction to link fraction
##############################################################################

def H_F_to_M(H,comp_pop_frac_tnsr):
    g_vec = comp_pop_frac_tnsr.shape
    comp_indices = make_composite_index(g_vec)
    M = np.zeros_like(H)
    pop_frac_vec = np.zeros(len(comp_indices))+np.nan
    for ivec in comp_indices:
        I = comp_index_to_integer(ivec,g_vec)
        pop_frac_vec[I] = comp_pop_frac_tnsr[ivec]
    pop_frac_vec = np.array(pop_frac_vec)
    for ivec in comp_indices:
        I = comp_index_to_integer(ivec,g_vec)
        norm = np.dot(pop_frac_vec,H[I,:])
        for jvec in comp_indices:
            J = comp_index_to_integer(jvec,g_vec)
            M[I,J] = comp_pop_frac_tnsr[ivec]*comp_pop_frac_tnsr[jvec]*H[I,J]/norm
    return M

def H_F_to_M_alternative(H,comp_pop_frac_tnsr):
    """
    Just a test
    """
    g_vec = comp_pop_frac_tnsr.shape
    comp_indices = make_composite_index(g_vec)
    M = np.zeros_like(H)
    pop_frac_vec = np.zeros(len(comp_indices))+np.nan
    for ivec in comp_indices:
        I = comp_index_to_integer(ivec,g_vec)
        pop_frac_vec[I] = comp_pop_frac_tnsr[ivec]
    pop_frac_vec = np.array(pop_frac_vec)
    for ivec in comp_indices:
        I = comp_index_to_integer(ivec,g_vec)
        # norm = np.dot(pop_frac_vec,H[I,:])
        for jvec in comp_indices:
            J = comp_index_to_integer(jvec,g_vec)
            M[I,J] = comp_pop_frac_tnsr[ivec]*comp_pop_frac_tnsr[jvec]*H[I,J]#/norm
    print (M)
    M = M/np.sum(M)
    return M

##############################################################################
##############################################################################
## EXPERIMENTS
##############################################################################
##############################################################################



def build_seed_disconn(comp_pop_frac_tnsr,m,directed):
    g_vec = comp_pop_frac_tnsr.shape
    comp_indices = make_composite_index(g_vec)
    nodes_per_group = int(np.ceil(1.0*m/len(comp_indices)))
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    node_attr_lst = []
    for cntr in range(nodes_per_group):
        i0 = cntr*len(comp_indices)
        node_attr_lst.extend([(i+i0,{"attr":i_vec}) for i, i_vec in enumerate(comp_indices)])
    G.add_nodes_from(node_attr_lst)
    return G

def build_seed_fullconn(comp_pop_frac_tnsr,m,directed):
    g_vec = comp_pop_frac_tnsr.shape
    comp_indices = make_composite_index(g_vec)
    nodes_per_group = int(np.ceil(1.0*m/len(comp_indices)))
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    node_attr_lst = []
    for cntr in range(nodes_per_group):
        i0 = cntr*len(comp_indices)
        node_attr_lst.extend([(i+i0,{"attr":i_vec}) for i, i_vec in enumerate(comp_indices)])
    G.add_nodes_from(node_attr_lst)
    for s,_ in node_attr_lst:
        for t,_ in node_attr_lst:
            if s!=t:
                G.add_edge(s,t)
    return G

def am(
    h_mtrx_lst,
    comp_pop_frac_tnsr,
    kind,
    directed=False,
    pop_fracs_lst = None,
    m=3,
    N=1000,
    G = None,
    v = 0,
    pa_on = False,
    ## Interaction-specific params
    alpha = None,
    p_d = None
    ):
    if directed and pa_on:
        raise Exception("Directed version of Fariba model with preferential attachment not correctly implemented. Currently not usable.")
    
    h_mtrx_lst = np.array(h_mtrx_lst)

    ## Assert that every parameter is within appropriate ranges
    am_preliminary_checks(
        h_mtrx_lst,
        comp_pop_frac_tnsr,
        pop_fracs_lst=pop_fracs_lst)

    ## Compute number of dimensions
    D = len(h_mtrx_lst)
    assert D == comp_pop_frac_tnsr.ndim

    ## Generate seed
    if G is None:
        G = build_seed_disconn(comp_pop_frac_tnsr,m,directed)

    ## Check I have enough nodes to make the initial m links
    assert m <= G.order()

    ## Make sure that nodes are adequately labeled for the
    ## model to work. If they are not contiguous integers
    ## and start at 0, the randint function to choose one 
    ## at random may fail. Using randint instead of random.choice 
    ## avoids having to grow nodelist and speeds up everything.
    nodelist = np.array(G.nodes())
    assert np.sum(np.abs(nodelist-np.arange(len(nodelist)))) < 1e-13

    ## Build probability distribution of composite populations
    memberships, probs = build_probs_pop(comp_pop_frac_tnsr)
    memberships_idx = list(range(len(memberships)))

    ## Build interaction function (faster than an if-switch inside 
    ## the inner for loop)
    if kind == "any":
        interaction = lambda h_vec,param: interaction_any(h_vec,param)
        param = None
    elif kind == "all":
        interaction = lambda h_vec,param: interaction_all(h_vec,param) 
        param = None
    elif kind == "one":
        assert len(p_d) == D
        interaction = lambda h_vec,param: interaction_one(h_vec,param)
        param = p_d
    elif kind == "max":
        interaction = lambda h_vec,param: interaction_max(h_vec,param)
        param = alpha
    elif kind == "min":
        interaction = lambda h_vec,param: interaction_min(h_vec,param)
        param = alpha
    elif kind == "hierarchy":
        interaction = lambda h_vec,param: interaction_hierarchy(h_vec,param)
    else:
        raise ValueError(f"Interaction kind {kind} invalid.")

    ## Iterate
    nI = G.order()
    nF = N+G.order()
    h_vec = np.zeros(D)
    if pa_on:
        degrees = np.zeros(nF)
        node_lst = np.arange(nF)
        for n, degi in G.in_degree():
            degrees[n] = degi
    for n in tqdm(range(nI,nF),disable=1-v):
        # if n%50 == 0 and v == 1:
            # print (p_deg[:50])
            # print (n)
#         assert n not in G ## If everything is correct, this can be commented for more speed
        ## Choose random type of node
        node_type_idx = np.random.choice(memberships_idx,p=probs)
        node_type = memberships[node_type_idx]
        ## Add it
        G.add_node(n,attr=node_type)
        ## Link it to m existing nodes
        n_lnks = 0
        while n_lnks < m:
            rnd_tie = False
            if pa_on:
                if n == nI:
                    target = np.random.randint(n)
                else:
                    rnd_tie = np.random.random()<0.15
                    if rnd_tie: ## Noise
                        target = np.random.randint(n)
                    else:
                        p_deg = degrees / np.sum(degrees)
                        target = np.random.choice(node_lst, p = p_deg)
            else:
                target = np.random.randint(n)
            
            ## Check if link exists
            if G.has_edge(n,target):
                continue
            
            ## Compute homophily
            orig_idx = node_type
            target_idx = G.nodes[target]["attr"]

            if kind == "hierarchy":
                h_vec = h_mtrx_lst
                param = (orig_idx,target_idx)
            else:
                for d in range(D):
                    h_vec[d] = h_mtrx_lst[
                            d,
                            orig_idx[d],
                            target_idx[d]
                            ]

            ## Check if the tie is made
            successful_tie = interaction(h_vec,param)

            ## Override if tie was made at random in PA regime
            # if pa_on:
            #     if rnd_tie:
            #         successful_tie = True

            ## Create links
            if successful_tie:
                G.add_edge(n,target)
                n_lnks+=1
                if pa_on:
                    degrees[target] += 1    
    return G





##############################################################################
## Measures of consolidation
##############################################################################

def my_MI(P):
    
    assert np.isclose(np.sum(P),1)

    Pi = np.sum(P,axis=1)
    Pj = np.sum(P,axis=0)
    MI = 0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i,j] != 0:
                MI += P[i,j]*np.log(P[i,j]/(Pi[i]*Pj[j]))
    return MI

##############################################################################
## Link counting metrics and utils
##############################################################################

def dir_mtrx_to_undir(dir_count_mtrx):

    # undir_mtrx = np.zeros_like(dir_count_mtrx)
    # n,m = dir_count_mtrx.shape
    # for i in range(n):
    #     undir_mtrx[i,i] = dir_count_mtrx[i,i]
    #     for j in range(i+1,m):
    #         undir_mtrx[i,j] = dir_count_mtrx[i,j] + dir_count_mtrx[j,i]

    undir_mtrx = dir_count_mtrx + dir_count_mtrx.T
    np.fill_diagonal(undir_mtrx, np.diag(undir_mtrx)/2.0)
    undir_mtrx_triu = np.triu(undir_mtrx)
    assert np.abs(np.sum(dir_count_mtrx) - np.sum(undir_mtrx_triu)) < 1e-13
    return undir_mtrx_triu

def counts_to_m(counts_mtrx):
    return counts_mtrx / (1.0*np.sum(counts_mtrx))

##############################################################################
## Simple inter-group link metrics
##############################################################################

def simple_inter_group_ties(count_mtrx,I_to_ivec,g_vec):
    ## 2022-10-25 changed input comp_pop_frac_tnsr by g_vec directly
    ## TO DO: Remove unused I_to_ivec parameter or implement some sort
    ## of test to assess the correctness of count_mtrx or something
    count_mtrx_lst = [np.zeros((g,g)) for g in g_vec]
    comp_indices = make_composite_index(g_vec)
    for ivec in comp_indices:
        I = comp_index_to_integer(ivec,g_vec)
        for jvec in comp_indices:
            J = comp_index_to_integer(jvec,g_vec)
            counts = count_mtrx[I,J]
            for d,i_d in enumerate(ivec):
                j_d = jvec[d]
                count_mtrx_lst[d][i_d,j_d] += counts
    return count_mtrx_lst

def all_simple_inter_group_to_m(count_mtrx,I_to_ivec,comp_pop_frac_tnsr):
    count_mtrx_lst = simple_inter_group_ties(count_mtrx,I_to_ivec,comp_pop_frac_tnsr)
    m_mtrx_lst = []
    for count_mtrx in count_mtrx_lst:
        m_mtrx_lst.append(counts_to_m(count_mtrx))
    return m_mtrx_lst

##############################################################################
## Composite inter-group link metrics
##############################################################################

def comp_inter_group_ties(
    G,
    comp_pop_frac_tnsr,
    # m
    ):
    g_vec = comp_pop_frac_tnsr.shape
    comp_indices = make_composite_index(g_vec)
    num_groups = len(comp_indices)
    count_mtrx = np.zeros((num_groups,num_groups))
    I_to_ivec = {comp_index_to_integer(ivec,g_vec):ivec for ivec in comp_indices}
    ## Simple inter-group counts - GIVES SAME RESULT AS THE OTHER METHOD (simple_inter_group_ties)
    # simple_counts = [np.zeros((g,g)) for g in g_vec]
    for e,(o,d) in enumerate(G.edges()):
        
        i_type = G.nodes[o]["attr"]
        j_type = G.nodes[d]["attr"]

        I = comp_index_to_integer(i_type,g_vec)
        J = comp_index_to_integer(j_type,g_vec)

        count_mtrx[I,J] += 1

        ## Simple inter-group counts - GIVES SAME RESULT AS THE OTHER METHOD (simple_inter_group_ties)
        # for d, i in enumerate(i_type):
            # j = j_type[d]
            # simple_counts[d][i,j] += 1

    # return count_mtrx, I_to_ivec, simple_counts - GIVES SAME RESULT AS THE OTHER METHOD (simple_inter_group_ties)
    return count_mtrx, I_to_ivec

##############################################################################
## Composite groups population counts (population abundance tensor)
##############################################################################

def comp_group_cnt_tnsr(pop_cnt_vec,g_vec):
    comp_indices = make_composite_index(g_vec)
    pop_cnt_tnsr = np.zeros(g_vec)
    pop_cnt_lst = [np.zeros(g) for g in g_vec]
    for ivec in comp_indices:
        I = comp_index_to_integer(ivec,g_vec)
        counts = pop_cnt_vec[I]
        pop_cnt_tnsr[tuple(ivec)] = counts
        for d, i_d in enumerate(ivec):
            pop_cnt_lst[d][i_d] += counts
    return pop_cnt_tnsr, pop_cnt_lst

def comp_group_cnt_tnsr_from_G(G,g_vec):
    pop_cnt_tnsr = np.zeros(g_vec)
    pop_cnt_lst = [np.zeros(g) for g in g_vec]
    for n in G:
        ivec = G.nodes[n]["attr"]
        pop_cnt_tnsr[tuple(ivec)] += 1
        for d, i_d in enumerate(ivec):
            pop_cnt_lst[d][i_d] += 1
    return pop_cnt_tnsr, pop_cnt_lst

def pop_cnt_tnsr_to_pop_cnts_1d_arr(pop_cnt_tnsr):
    g_vec = pop_cnt_tnsr.shape
    comp_indices = make_composite_index(g_vec)
    cnts_1d_arr = []
    for ivec in comp_indices:
        cnts_1d_arr.append(pop_cnt_tnsr[ivec])
    return np.array(cnts_1d_arr)

##############################################################################
##############################################################################
## FITTING THE GROWTH MODEL
##############################################################################
##############################################################################

def build_2d2a_h_mtrx(h0_00, h0_11, h1_00, h1_11):
    h_mtrx_lst = [np.zeros((2,2)) for _ in range(2)]

    h_mtrx_lst[0][0,0] = h0_00
    h_mtrx_lst[0][0,1] = 1.0-h0_00
    h_mtrx_lst[0][1,0] = 1.0-h0_11
    h_mtrx_lst[0][1,1] = h0_11

    h_mtrx_lst[1][0,0] = h1_00
    h_mtrx_lst[1][0,1] = 1.0-h1_00
    h_mtrx_lst[1][1,0] = 1.0-h1_11
    h_mtrx_lst[1][1,1] = h1_11
    return h_mtrx_lst

def build_2d2a_h_mtrx_hierarchy(x_h):
    h_mtrx_lst = [np.zeros((2,2)), np.zeros((2,2,2))]

    h_mtrx_lst[0][0,0] = x_h[0]
    h_mtrx_lst[0][0,1] = 1.0-x_h[0]
    h_mtrx_lst[0][1,0] = 1.0-x_h[1]
    h_mtrx_lst[0][1,1] = x_h[1]

    h_mtrx_lst[1][0,0,0] = x_h[2]
    h_mtrx_lst[1][0,0,1] = 1.0-x_h[2]
    h_mtrx_lst[1][0,1,0] = 1.0-x_h[3]
    h_mtrx_lst[1][0,1,1] = x_h[3]

    h_mtrx_lst[1][1,0,0] = x_h[4]
    h_mtrx_lst[1][1,0,1] = 1.0-x_h[4]
    h_mtrx_lst[1][1,1,0] = 1.0-x_h[5]
    h_mtrx_lst[1][1,1,1] = x_h[5]

    return h_mtrx_lst

def build_2d2a_h_mtrx_v2(x_h):
    if len(x_h) == 4:
        h_mtrx_lst = build_2d2a_h_mtrx(*tuple(x_h))
    elif len(x_h) == 6:
        h_mtrx_lst = build_2d2a_h_mtrx_hierarchy(x_h)
    else:
        raise ValueError(f"x_h {x_h} does not have appropriate length.")
    return h_mtrx_lst

def model_2_dim_2_attr_UD(
    x,
    comp_pop_frac_tnsr,
    kind
    ):

    ## Homophily and additional parameters to optimize
    if kind == "any":
        # h0_00, h0_11, h1_00, h1_11 = x
        x_h = x
        p_d = None
        alpha = None
    elif kind == "one":
        p_d = [0,0]
        x_h = x[:4]
        # h0_00, h0_11, h1_00, h1_11, p_d[0] = x
        p_d[0] = x[4]
        p_d[1] = 1.0-p_d[0]
        alpha = None
    elif kind == "max":
        # h0_00, h0_11, h1_00, h1_11, alpha = x
        x_h = x[:4]
        alpha = x[4]
        p_d = None
    elif kind == "min":
        # h0_00, h0_11, h1_00, h1_11, alpha = x
        x_h = x[:4]
        alpha = x[4]
        p_d = None
    elif kind == "hierarchy":
        # h0_00, h0_11, h10_00, h10_11, h11_00, h11_11 = x
        x_h = x
        alpha = None
        p_d = None
    else:
        raise ValueError(f"Interaction kind {kind} invalid.")

    ## Build homophily matrices for each dimension
    # h_mtrx_lst = build_2d2a_h_mtrx(h0_00, h0_11, h1_00, h1_11)
    h_mtrx_lst = build_2d2a_h_mtrx_v2(x_h)

    ## Theoretical values of inter-group links
    H_theor = composite_H(
        h_mtrx_lst,
        kind,
        p_d = p_d,
        alpha = alpha,
        )
    M_theor = H_F_to_M(H_theor,comp_pop_frac_tnsr)
    M_theor_ud = dir_mtrx_to_undir(M_theor)

    return M_theor_ud,H_theor,h_mtrx_lst

def model_2_dim_2_attr_UD_resid(x,
    M_emp_ud,
    comp_pop_frac_tnsr,
    kind):

    M_theor_ud,_,_ = model_2_dim_2_attr_UD(
        x,
        comp_pop_frac_tnsr,
        kind
        )

    M_theor_ud_vec = M_theor_ud[np.triu_indices(len(M_theor_ud))]
    M_emp_ud_vec = M_emp_ud[np.triu_indices(len(M_emp_ud))]

    return (M_theor_ud_vec-M_emp_ud_vec)

def fit_model_2_dim_2_attr_UD(
    M_mtrx_ud,
    comp_pop_frac_tnsr,
    kind,
    n_params,
    n_annealing=10,
    n_ls=100,
    n_basin=10,
    ):
    ## Loss function is sum of square of residuals
    optim_func = lambda x:0.5*np.sum(model_2_dim_2_attr_UD_resid(x,M_mtrx_ud,comp_pop_frac_tnsr,kind)**2.0)

    print ("Dual annealing")
    ann_res = []
    for i in range(n_annealing):
        print (i)
        ann_res_i = optimize.dual_annealing(
            optim_func,
            bounds= [(0,1)]*n_params,
            maxiter=2000,
            x0 = np.random.random(size=n_params)
            )
        print (ann_res_i["message"],ann_res_i["success"],ann_res_i["fun"])
        ann_res.append((ann_res_i["x"],ann_res_i["fun"]))
    print ("**********************************")

    print ("NL least squares")
    ls_res = []
    for comp in range(n_ls):
        print (comp)
        x0 = np.random.random(size=n_params)
        x_res = optimize.least_squares(
            model_2_dim_2_attr_UD_resid,
            x0,
            bounds=(0,1),
            jac = "3-point",
            ftol = 1e-15,
            xtol = 1e-15,
            gtol = 1e-15,
            diff_step = 1e-15,
        #     loss = "arctan",
            args=(M_mtrx_ud,
                  comp_pop_frac_tnsr,
                  kind)
            )
        print (x_res["cost"])
        ls_res.append((x_res["x"],x_res["cost"]))
    print ("***********************************")

    print ("Basin hopping")
    basin_res = []
    for i in range(n_basin):
        print (i)
        basin_res_i = optimize.differential_evolution(
            optim_func,
            x0 = np.random.random(size=n_params),
            bounds= [(0,1)]*n_params,
            )
        print (basin_res_i["message"],basin_res_i["success"],basin_res_i["fun"])
        print (basin_res_i)
        basin_res.append((basin_res_i["x"],basin_res_i["fun"]))
    print ("***********************************")

    try:
        ann_min = min(ann_res, key=lambda x:x[1]) 
    except ValueError:
        ann_min = None

    try:
        ls_min = min(ls_res, key=lambda x:x[1]) 
    except ValueError:
        ls_min = None

    try:
        basin_min = min(basin_res, key=lambda x:x[1]) 
    except ValueError:
        basin_min = None

    return ann_res,basin_res,ls_res,dict([("annealing",ann_min),("basin",basin_min),("LS",ls_min)])


##############################################################################
##############################################################################
## FITTING THE E-R MODEL
##############################################################################
##############################################################################

##############################################################################
## 1D Theoretical number of inter-group links (for tests)
##############################################################################

def ER_1D_inter_group_links(h_mtrx,pop_fracs):
    h_mtrx = np.array(h_mtrx)
    assert h_mtrx.shape[0] == h_mtrx.shape[1]
    m0 = np.zeros_like(h_mtrx)
    for i in range(h_mtrx.shape[0]):
        for j in range(h_mtrx.shape[0]):
            hij = h_mtrx[i,j]
            fi = pop_fracs[i]
            fj = pop_fracs[j]
            m0[i,j] = hij*fi*fj
    return m0 / np.sum(m0)

def ER_1D_solve_h_mtrx_BY_HAND(M_mtrx_dir,pop_fracs):
    assert M_mtrx_dir.shape[0] == M_mtrx_dir.shape[1]
    assert np.abs(np.sum(M_mtrx_dir) - 1) < 1e-14
    assert np.abs(np.sum(pop_fracs) - 1) < 1e-14
    A1 = np.zeros((2,2))
    A2 = np.zeros((2,2))
    
    fa,fb = pop_fracs
    
    a_g_a = M_mtrx_dir[0,0] / (M_mtrx_dir[0,0] + M_mtrx_dir[0,1])
    b_g_a = M_mtrx_dir[0,1] / (M_mtrx_dir[0,0] + M_mtrx_dir[0,1])
    
    A1[0,0] = (a_g_a*fa**2.0-fa**2.0)
    A1[0,1] = (a_g_a*fa*fb)
    
    A1[1,0] = (b_g_a*fa**2.0)
    A1[1,1] = (b_g_a*fa*fb-fa*fb)
    
    b_g_b = M_mtrx_dir[1,1] / (M_mtrx_dir[1,0] + M_mtrx_dir[1,1])
    a_g_b = M_mtrx_dir[1,0] / (M_mtrx_dir[1,0] + M_mtrx_dir[1,1])
    
    A2[0,0] = (b_g_b*fb**2.0-fb**2.0)
    A2[0,1] = (b_g_b*fa*fb)
    
    A2[1,0] = (a_g_b*fb**2.0)
    A2[1,1] = (a_g_b*fa*fb-fa*fb)
    
    
    ########
    ## Option 2 (with SVD, more numerically stable?)
    _,vals,vecs = mp.svd_r(mp.matrix(A1))
    sol_ind = np.argwhere(np.abs(vals)<1e-16)
    if len(sol_ind) != 1:
        print (M_mtrx_dir,pop_fracs,A,vals)
        return None 
    if not (np.all(np.array(vecs[sol_ind[0,0],:])<0) or np.all(np.array(vecs[sol_ind[0,0],:]) >0)):
        print (M_mtrx_dir,pop_fracs,A,vals,vecs[sol_ind[0,0],:])
        return None
    res1 = np.abs(vecs[sol_ind[0,0],:])
    ########

    ########
    ## Option 2 (with SVD, more numerically stable?)
    _,vals,vecs = mp.svd_r(mp.matrix(A2))
    sol_ind = np.argwhere(np.abs(vals)<1e-16)
    if len(sol_ind) != 1:
        print (M_mtrx_dir,pop_fracs,A,vals)
        return None 
    if not (np.all(np.array(vecs[sol_ind[0,0],:])<0) or np.all(np.array(vecs[sol_ind[0,0],:]) >0)):
        print (M_mtrx_dir,pop_fracs,A,vals,vecs[sol_ind[0,0],:])
        return None
    res2 = np.abs(vecs[sol_ind[0,0],:])
    ########
    
    return res1, res2

def ER_1D_solve_h_mtrx_dens(M_cnts_dir,pop_cnts,get_CI = False):
#     E = np.sum(M_cnts_dir)
#     N = np.sum(pop_cnts)
#     rho = E/N**2.0 ## Approximation of E / (N (N-1))

    h_mtrx = np.zeros_like(M_cnts_dir)
    if get_CI:
        h_mtrx_u = np.zeros_like(M_cnts_dir)
        h_mtrx_l = np.zeros_like(M_cnts_dir)
    g = M_cnts_dir.shape[0]
    for i in range(g):
        for j in range(g):
            if pop_cnts[i]*pop_cnts[j] == 0:
                ## Enforce h_mtrx[i,j] = 0 instead of np.nan when it is 0/0
                h_mtrx[i,j] = 0
            else:
                
                h_mtrx[i,j] = M_cnts_dir[i,j] / (pop_cnts[i]*pop_cnts[j])
                
                if get_CI:
                    l,u = proportion_confint(
                        M_cnts_dir[i,j],
                        pop_cnts[i]*pop_cnts[j],
                        method="wilson")
                    h_mtrx_l[i,j] = l
                    h_mtrx_u[i,j] = u
    
    if get_CI:
        return h_mtrx, h_mtrx_l, h_mtrx_u
    else:
        return h_mtrx

def ER_1D_solve_h_mtrx(M_mtrx_dir,pop_fracs):
    assert M_mtrx_dir.shape[0] == M_mtrx_dir.shape[1]
    assert np.abs(np.sum(M_mtrx_dir) - 1) < 1e-14
    assert np.abs(np.sum(pop_fracs) - 1) < 1e-14
    g = M_mtrx_dir.shape[0]
    A = np.zeros((g**2,g**2))

    for i in range(g**2):
        p1 = i // 2
        q1 = i % 2
        mpq = M_mtrx_dir[p1,q1]
        for j in range(g**2):
            p2 = j // 2 
            q2 = j % 2
            if i == j:
                A[i,j] = mpq*pop_fracs[p2]*pop_fracs[q2] - pop_fracs[p2]*pop_fracs[q2]
            else:
                A[i,j] = mpq*pop_fracs[p2]*pop_fracs[q2]

    ########
    ## Option 1 (with eig)
    ## Eigenvectors of AA' (SVD) - the eigval=0 has the solution to the 
    ## homogeneous linear system
#     vals, vecs = np.linalg.eigh(np.dot(A.T, A))
#     vals, vecs = mp.eig(mp.matrix(A.T)*mp.matrix(A))
#     sol_ind = np.argwhere(np.abs(vals)<1e-15)
#     assert len(sol_ind) == 1
#     assert np.all(np.array(vecs[:,sol_ind[0,0]])<0) or np.all(np.array(vecs[:,sol_ind[0,0]]) >0)
#     res = np.abs(vecs[:,sol_ind[0,0]])
    ########
    
    ########
    ## Option 2 (with SVD, more numerically stable?)
    _,vals,vecs = mp.svd_r(mp.matrix(A))
    sol_ind = np.argwhere(np.abs(vals)<1e-16)
    if len(sol_ind) != 1:
        print (M_mtrx_dir,pop_fracs,A,vals)
        return np.zeros_like(A)+np.nan,np.zeros(A.shape[0]) + np.nan 
    if not (np.all(np.array(vecs[sol_ind[0,0],:])<0) or np.all(np.array(vecs[sol_ind[0,0],:]) >0)):
        print (M_mtrx_dir,pop_fracs,A,vals,vecs[sol_ind[0,0],:])
        return np.zeros_like(A)+np.nan,np.zeros(A.shape[0]) + np.nan
    res = np.abs(vecs[sol_ind[0,0],:])
    ########

    return A, res

def ER_1D_fit_h_mtrx(M_mtrx_dir,pop_fracs,
    n_params=4,
    n_annealing=10,
    n_ls=100,
    n_basin=10,
    x0 = None):
    assert np.abs(np.sum(M_mtrx_dir) - 1.0) < 1e-13

    ## Loss function is sum of square of residuals
    optim_func = lambda x:0.5*np.sum((ER_1D_inter_group_links(np.array([ [x[0],x[1]],[x[2],x[3]] ]),pop_fracs)-M_mtrx_dir)**2.0)

    if x0 is None:
        x0_rnd = True
    else:
        x0_rnd = False
    
#     print ("Dual annealing")
#     ann_res = []
#     for i in range(n_annealing):
#         print (i)
#         if x0_rnd:
#             x0 = np.random.random(size=n_params)
#         ann_res_i = optimize.dual_annealing(
#             optim_func,
#             bounds= [(0,1)]*n_params,
#             maxiter=2000,
#             x0 = x0
#             )
#         print (ann_res_i["message"],"success",ann_res_i["success"],ann_res_i["fun"],ann_res_i["x"])
#         ann_res.append((ann_res_i["x"],ann_res_i["fun"]))
#     print ("**********************************")

    print ("Basin hopping")
    basin_res = []
    for i in range(n_basin):
        print (i)
        if x0_rnd:
            x0 = np.random.random(size=n_params)
        basin_res_i = optimize.differential_evolution(
            optim_func,
            x0 = x0,
            tol = 1e-40,
            atol= 1e-40,
            bounds= [(0,1)]*n_params,
            )
        print (basin_res_i["message"],"success",basin_res_i["success"],basin_res_i["fun"],basin_res_i["x"])
        print (basin_res_i)
        basin_res.append((basin_res_i["x"],basin_res_i["fun"]))
    print ("***********************************")
    
##############################################################################
## E-R N-dimensional theoretical number of inter-group links (for tests)
##############################################################################

def ER_inter_group_links_theor(
    h_mtrx_lst,
    comp_pop_frac_tnsr,
    kind,
    p_d = None,
    alpha = None
    ):
    H_theor = composite_H(
        h_mtrx_lst,
        kind,
        p_d = p_d,
        alpha = alpha,
        )

    g_vec = comp_pop_frac_tnsr.shape

    comp_indices = make_composite_index(g_vec)
    assert len(comp_indices[0]) == len(h_mtrx_lst)

    G = len(comp_indices)
    M_theor = np.zeros((G,G))+np.nan
    for i_vec in comp_indices:
        I = comp_index_to_integer(i_vec,g_vec)
        for j_vec in comp_indices:
            J = comp_index_to_integer(j_vec,g_vec)
            F_I = comp_pop_frac_tnsr[tuple(i_vec)]
            F_J = comp_pop_frac_tnsr[tuple(j_vec)]
            M_theor[I,J] = F_I*F_J*H_theor[I,J]

    return M_theor/np.sum(M_theor), H_theor

##############################################################################
## Fitting the model with least squares
##############################################################################

def build_2d2a_h_mtrx_ER_DIR(x):
    assert len(x) == 8
    h_mtrx_lst = [x[:4].reshape((2,2)),x[4:].reshape((2,2))]
    return h_mtrx_lst

def model_2_dim_2_attr_ER_DIR(
    x,
    # comp_pop_frac_tnsr,
    kind
    ):

    ## Homophily and additional parameters to optimize
    if kind in ["all","any"]:
        # h0_00, h0_11, h1_00, h1_11 = x
        x_h = x
        p_d = None
        alpha = None
    elif kind == "one":
        p_d = [0,0]
        x_h = x[:4]
        # h0_00, h0_11, h1_00, h1_11, p_d[0] = x
        p_d[0] = x[4]
        p_d[1] = 1.0-p_d[0]
        alpha = None
    elif kind == "max":
        # h0_00, h0_11, h1_00, h1_11, alpha = x
        x_h = x[:4]
        alpha = x[4]
        p_d = None
    elif kind == "min":
        # h0_00, h0_11, h1_00, h1_11, alpha = x
        x_h = x[:4]
        alpha = x[4]
        p_d = None
    elif kind == "hierarchy":
        # h0_00, h0_11, h10_00, h10_11, h11_00, h11_11 = x
        x_h = x
        alpha = None
        p_d = None
    else:
        raise ValueError(f"Interaction kind {kind} invalid.")

    ## Build homophily matrices for each dimension
    # h_mtrx_lst = build_2d2a_h_mtrx(h0_00, h0_11, h1_00, h1_11)
    h_mtrx_lst = build_2d2a_h_mtrx_ER_DIR(x_h)

    ## Theoretical values of inter-group links
    # M_theor = ER_inter_group_links_theor(
    #     h_mtrx_lst,
    #     comp_pop_frac_tnsr,
    #     kind,
    #     p_d = None,
    #     alpha = None
    #     )

    H_theor = composite_H(
        h_mtrx_lst,
        kind,
        p_d = p_d,
        alpha = alpha,
        )

    return H_theor

def model_2_dim_2_attr_ER_DIR_resid(
    x,
    H_infer,
    kind):

    H_theor = model_2_dim_2_attr_ER_DIR(
        x,
        # comp_pop_frac_tnsr,
        kind
        )

    return (H_theor[~np.isnan(H_infer)]-H_infer[~np.isnan(H_infer)]).ravel() ## This ~np.isnan mask prevents UNKNOWN empirical H values to have any effect on the inference

def fit_model_2_dim_2_attr_ER_DIR(
    M_cnts_dir,
    pop_cnts,
    kind,
    n_params,
    n_annealing=10,
    n_ls=100,
    n_basin=10,
    ):
    H_infer = ER_1D_solve_h_mtrx_dens(M_cnts_dir,pop_cnts)

    ## Loss function is sum of square of residuals
    optim_func = lambda x:0.5*np.sum(model_2_dim_2_attr_ER_DIR_resid(x,H_infer,kind)**2.0)

    print ("Dual annealing")
    ann_res = []
    for i in range(n_annealing):
        print (i)
        ann_res_i = optimize.dual_annealing(
            optim_func,
            bounds= [(0,1)]*n_params,
            maxiter=2000,
            x0 = np.random.random(size=n_params)
            )
        print (ann_res_i["message"],ann_res_i["success"],ann_res_i["fun"],ann_res_i["x"])
        ann_res.append((ann_res_i["x"],ann_res_i["fun"]))
    print ("**********************************")

    print ("NL least squares")
    ls_res = []
    for comp in range(n_ls):
        print (comp)
        success_ls = False
        while not success_ls: ## To deal with error "residuals are not finite at initial point"
            x0 = np.random.random(size=n_params)
            try:
                x_res = optimize.least_squares(
                    model_2_dim_2_attr_ER_DIR_resid,
                    x0,
                    bounds=(0,1),
                    jac = "3-point",
                    ftol = 1e-15,
                    xtol = 1e-15,
                    gtol = 1e-15,
                    diff_step = 1e-15,
                #     loss = "arctan",
                    args=(H_infer,
                          kind)
                    )
                success_ls = True
            except ValueError:
                print (x0, H_infer)
                continue
        print (x_res["cost"],x_res["x"])
        ls_res.append((x_res["x"],x_res["cost"]))
    print ("***********************************")

    print ("Basin hopping")
    basin_res = []
    for i in range(n_basin):
        print (i)
        basin_res_i = optimize.differential_evolution(
            optim_func,
            x0 = np.random.random(size=n_params),
            bounds= [(0,1)]*n_params,
            )
        print (basin_res_i["message"],basin_res_i["success"],basin_res_i["fun"],basin_res_i["x"])
        print (basin_res_i)
        basin_res.append((basin_res_i["x"],basin_res_i["fun"]))
    print ("***********************************")

    try:
        ann_min = min(ann_res, key=lambda x:x[1]) 
    except ValueError:
        ann_min = None

    try:
        ls_min = min(ls_res, key=lambda x:x[1]) 
    except ValueError:
        ls_min = None

    try:
        basin_min = min(basin_res, key=lambda x:x[1]) 
    except ValueError:
        basin_min = None

    return ann_res,basin_res,ls_res,dict([("annealing",ann_min),("basin",basin_min),("LS",ls_min)])


##############################################################################
## ER model arbitrary number of dimensions
##############################################################################

def build_NdNa_h_mtrx_ER_DIR(x,g_vec):
    # print ("REVISAR BIEN ESTA FUNCION QUE IGUAL ESTA MALA. MIRAR EL TRY EXCEPT DE LA DE ABAJO TAMBIEN.")

    ## Verification
    n_param = 0
    for v_d in g_vec:
        n_param += v_d**2
    assert len(x) == n_param

    ## Build h matrix list
    h_mtrx_lst = []
    start_ind = 0
    for v_d in g_vec:
        h_mtrx = x[start_ind:start_ind+v_d**2].reshape((v_d,v_d))
        h_mtrx_lst.append(h_mtrx)
        start_ind += v_d**2
    return h_mtrx_lst

def model_N_dim_N_attr_ER_DIR(
    x,
    g_vec,
    kind
    ):

    D = len(g_vec)
    ## Homophily and additional parameters to optimize
    if kind in ["all","any"]:
        # h0_00, h0_11, h1_00, h1_11 = x
        x_h = x
        p_d = None
        alpha = None
    elif kind == "one":
        x_h = x[:D]
        # h0_00, h0_11, h1_00, h1_11, p_d[0] = x
        p_d = x[D:]
        alpha = None
    elif kind == "max":
        # h0_00, h0_11, h1_00, h1_11, alpha = x
        x_h = x[:-1]
        alpha = x[-1]
        p_d = None
    elif kind == "min":
        # h0_00, h0_11, h1_00, h1_11, alpha = x
        x_h = x[:-1]
        alpha = x[-1]
        p_d = None
    elif kind == "hierarchy":
        # h0_00, h0_11, h10_00, h10_11, h11_00, h11_11 = x
        raise ValueError(f"Interaction kind {kind} not yet implemented.")
        # x_h = x
        # alpha = None
        # p_d = None
    else:
        raise ValueError(f"Interaction kind {kind} invalid.")

    ## Build homophily matrices for each dimension
    h_mtrx_lst = build_NdNa_h_mtrx_ER_DIR(x_h, g_vec)

    H_theor = composite_H(
        h_mtrx_lst,
        kind,
        p_d = p_d,
        alpha = alpha,
        )

    return H_theor

def model_N_dim_N_attr_ER_DIR_resid(
    x,
    H_infer,
    g_vec,
    kind):
    
    H_theor = model_N_dim_N_attr_ER_DIR(
        x,
        g_vec,
        kind
        )

    return (H_theor[~np.isnan(H_infer)]-H_infer[~np.isnan(H_infer)]).ravel() ## This ~np.isnan mask prevents UNKNOWN empirical H values to have any effect on the inference

def fit_model_N_dim_N_attr_ER_DIR(
    M_cnts_dir,
    pop_cnts,
    g_vec,
    kind,
    n_params,
    # n_annealing=10,
    n_ls=1,
    # n_basin=10,
    ):

    ## Verification
    G = 1
    for v_d in g_vec:
        G *= v_d
    assert G == len(pop_cnts)

    H_infer = ER_1D_solve_h_mtrx_dens(M_cnts_dir,pop_cnts)

    ## Loss function is sum of square of residuals
    optim_func = lambda x:0.5*np.sum(model_N_dim_N_attr_ER_DIR_resid(x,H_infer,g_vec,kind)**2.0)

    ls_res = []
    for comp in range(n_ls):
        print (comp)
        success_ls = False
        while not success_ls: ## To deal with error "residuals are not finite at initial point"
            x0 = np.random.random(size=n_params)
            try:
                x_res = optimize.least_squares(
                    model_N_dim_N_attr_ER_DIR_resid,
                    x0,
                    bounds=(0,1),
                    jac = "3-point",
                    ftol = 1e-15,
                    xtol = 1e-15,
                    gtol = 1e-15,
                    diff_step = 1e-15,
                #     loss = "arctan",
                    args=(H_infer,
                        g_vec,
                        kind)
                    )
                success_ls = True
            except ValueError:
                print (x0, H_infer)
                continue
        print (x_res["cost"],x_res["x"])
        ls_res.append((x_res["x"],x_res["cost"]))

    try:
        ls_min = min(ls_res, key=lambda x:x[1]) 
    except ValueError:
        ls_min = None

    return ls_res, ls_min

