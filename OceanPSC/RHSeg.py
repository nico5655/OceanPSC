import numpy as np
import operations as op

def get_frequencies(data):
    dist=5
    freqs=np.zeros((35,data.shape[0],data.shape[1]))
    for k in range(32):
        freqs[k]=divide_treatment(data,lambda data: (op.neighbor_grid(data,dist)==k).mean(axis=(-1,-2)))

    def LBP(data,R):
        filtre=np.ones((2*R+1,2*R+1))
        filtre[1:-1,1:-1]=0
        edit=filtre[filtre==1]
        for k in range(1,len(edit)+1):
            edit[k-1]=k
        filtre[filtre==1]=edit
        vois=op.neighbor_grid(data,R)
        sval=(vois.T==data.T).T
        return np.sum((filtre>=1)*(sval)*2**filtre,axis=(-1,-2))

    freqs[-3]=divide_treatment(data, lambda d: LBP(d,1))
    freqs[-2]=divide_treatment(data, lambda d: LBP(d,2))
    freqs[-1]=divide_treatment(data, lambda d: LBP(d,3))

    return freqs.transpose(1,2,0)


def rec_hseg(w_data,level):
    ### result variables
    if level==max_level:
        stop_regions=min_regions
    else:
        stop_regions=min_regions//2
    if level == 1:
        cl_rslt=np.arange(w_data.shape[0]*w_data.shape[1]).reshape(w_data.shape[0],w_data.shape[1])
        region_caracs=w_data.reshape(-1,35)
        region_sizes=np.int32(np.ones(cl_rslt.shape).reshape(-1))
    else:
        cl_rslt=np.zeros((w_data.shape[0],w_data.shape[1]))
        #the 4 part split
        vmid=w_data.shape[0]//2
        hmid=w_data.shape[1]//2
        cl1,rc1,rs1=rec_hseg(w_data[:vmid,:hmid],level-1)
        cl_rslt[:vmid,:hmid]=cl1
        region_caracs=rc1
        region_sizes=rs1
        cl2,rc2,rs2=rec_hseg(w_data[:vmid,hmid:],level-1)
        cl_rslt[:vmid,hmid:]=(cl2+np.max(cl_rslt))+1
        region_caracs+=rc2
        region_sizes+=rs2
        cl3,rc3,rs3=rec_hseg(w_data[vmid:,:hmid],level-1)
        cl_rslt[vmid:,:hmid]=(cl3+np.max(cl_rslt))+1
        region_caracs+=rc3
        region_sizes+=rs3
        cl4,rc4,rs4=rec_hseg(w_data[vmid:,hmid:],level-1)
        cl_rslt[vmid:,hmid:]=(cl4+np.max(cl_rslt))+1
        region_caracs+=rc4
        region_sizes+=rs4
        region_caracs=np.array(region_caracs)
        region_sizes=np.array(region_sizes)

    ### intermediary variables
    if level==1:
        all_regions=list(cl_rslt.copy().reshape(-1))
    else:
        all_regions=[k for k in range(len(region_sizes))]

    si=int(np.max(all_regions))+1
    dists=np.zeros((si,si))
    adjacents=np.zeros((si,si))
    distincts=np.ones((si,si))
    for l in range(si):
        distincts[l,l]=0

    ### internal functions

    def H(n,X):
        rs=-X*np.log(X+(X==0))#0*log(0)=0 enforced by adding 1 to 0 values in log
        return n*np.sum(rs,axis=-1)

    def all_dists(i):
        def H(n,X):
            rs=-X*np.log(X+(X==0))#0*log(0)=0 enforced by adding 1 to 0 values in log
            return n*np.sum(rs,axis=-1)
        itr=(region_caracs.T*region_sizes).T+region_caracs[i]*region_sizes[i]
        merged=((itr.T)/(region_sizes+region_sizes[i])).T
        return H(region_sizes+region_sizes[i],merged)-(H(region_sizes[i],region_caracs[i])+H(region_sizes,region_caracs))

    def merge(i,j):
        if (not j in all_regions) or (not i in all_regions):
            return
        if i>j:
            return merge(j,i)
        region_caracs[i]=(region_caracs[i]*region_sizes[i]+region_caracs[j]*region_sizes[j])/(region_sizes[i]+region_sizes[j])
        region_sizes[i]+=region_sizes[j]
        region_sizes[j]=0
        cl_rslt[cl_rslt==j]=i
        all_regions.remove(j)
        vois=op.neighbor_grid(cl_rslt,bound_method='duplicate')
        
            
        for k in all_regions:
            if k!=i and adjacents[k,j]:
                adjacents[i,k]=True
                adjacents[k,i]=True
        
        adjacents[j,:]=False
        adjacents[:,j]=False

        a_dists=all_dists(i)
        dists[:,i]=a_dists
        dists[i,:]=a_dists

    ### init
    vois=op.neighbor_grid(cl_rslt,bound_method='duplicate')
    for i in all_regions:
        dta=np.int32(np.unique(vois[cl_rslt==i]))
        for j in dta:
            if j!=i:
                adjacents[i,j]=True
        a_dists=all_dists(i)
        for k in all_regions:
            d=a_dists[k]
            dists[i,k]=d
            dists[k,i]=d

    ### treatment
    while len(all_regions)>stop_regions:
        min_h=np.min(dists[adjacents==1])
        for reg in all_regions:
            for j in all_regions:
                if dists[reg,j]==min_h and adjacents[reg,j]==1:
                    merge(reg,j)
        
        pred=(distincts==1)&(dists<=min_h)
        if pred.sum()>0:
            min_h2=np.min(dists[pred])
            for reg in all_regions:
                    for j in all_regions:
                        if j>reg:
                            cmh=dists[reg,j]
                            if cmh<=min_h2:
                                merge(reg,j)

    ### result cleaning
    n_regions_caracs=[]
    n_regions_sizes=[]
    #random.shuffle(all_regions)
    for k in range(len(all_regions)):
        reg=all_regions[k]
        n_regions_caracs.append(region_caracs[reg])
        n_regions_sizes.append(region_sizes[reg])
        cl_rslt[cl_rslt==reg]=k
    if level == max_level-1:
        print('1 quarter')
    return cl_rslt,n_regions_caracs,n_regions_sizes

min_regions=512
convfact=1.005
critval=0
max_level=6

if __name__=='main':
    r_cl=np.load('data/r_cl.npy')
    god_power=get_frequencies(block_reduce(r_cl,(3,3),np.median))
    freqs=god_power.transpose(2,0,1)
    red_freqs=block_reduce(freqs,(1,3,3),np.mean)
    big_data=red_freqs.transpose(1,2,0)
    big_data=big_data[2:-2,:]

    t=time.time()
    rst=rec_hseg(big_data,max_level)
    print(time.time()-t)
    np.save('data/prem.npy',rst[0])

    cl_rslt=rst[0].copy()
    region_caracs=np.array(rst[1].copy())
    region_sizes=np.array(rst[2].copy())

    all_regions=[k for k in range(len(region_sizes))]
    si=int(np.max(all_regions))+1
    dists=np.zeros((si,si))
    adjacents=np.zeros((si,si))
    distincts=np.ones((si,si))
    for l in range(si):
        distincts[l,l]=0

    ### internal functions

    def H(n,X):
        rs=-X*np.log(X+(X==0))#0*log(0)=0 enforced by adding 1 to 0 values in log
        return n*np.sum(rs,axis=-1)

    def all_dists(i):
        def H(n,X):
            rs=-X*np.log(X+(X==0))#0*log(0)=0 enforced by adding 1 to 0 values in log
            return n*np.sum(rs,axis=-1)
        itr=(region_caracs.T*region_sizes).T+region_caracs[i]*region_sizes[i]
        merged=((itr.T)/(region_sizes+region_sizes[i])).T
        return H(region_sizes+region_sizes[i],merged)-(H(region_sizes[i],region_caracs[i])+H(region_sizes,region_caracs))

    def merge(i,j):
        if (not j in all_regions) or (not i in all_regions):
            return
        if i>j:
            return merge(j,i)
        region_caracs[i]=(region_caracs[i]*region_sizes[i]+region_caracs[j]*region_sizes[j])/(region_sizes[i]+region_sizes[j])
        region_sizes[i]+=region_sizes[j]
        region_sizes[j]=0
        cl_rslt[cl_rslt==j]=i
        all_regions.remove(j)
        vois=op.neighbor_grid(cl_rslt,bound_method='duplicate')


        for k in all_regions:
            if k!=i and adjacents[k,j]:
                adjacents[i,k]=True
                adjacents[k,i]=True

        adjacents[j,:]=False
        adjacents[:,j]=False

        a_dists=all_dists(i)
        dists[:,i]=a_dists
        dists[i,:]=a_dists

    ### init
    vois=op.neighbor_grid(cl_rslt,bound_method='duplicate')
    for i in all_regions:
        dta=np.int32(np.unique(vois[cl_rslt==i]))
        for j in dta:
            if j!=i:
                adjacents[i,j]=True
        a_dists=all_dists(i)
        for k in all_regions:
            d=a_dists[k]
            dists[i,k]=d
            dists[k,i]=d

    ### treatment
    while len(all_regions)>32:
        min_h=np.min(dists[adjacents==1])
        for reg in all_regions:
            for j in all_regions:
                if dists[reg,j]==min_h and adjacents[reg,j]==1:
                    merge(reg,j)

        pred=(distincts==0)&(dists<=min_h)
        if pred.sum()>0:
            min_h2=np.min(dists[pred])
            for reg in all_regions:
                    for j in all_regions:
                        if j>reg:
                            cmh=dists[reg,j]
                            if cmh<=min_h2:
                                merge(reg,j)

    ### result cleaning
    n_regions_caracs=[]
    n_regions_sizes=[]
    #random.shuffle(all_regions)
    for k in range(len(all_regions)):
        reg=all_regions[k]
        n_regions_caracs.append(region_caracs[reg])
        n_regions_sizes.append(region_sizes[reg])
        cl_rslt[cl_rslt==reg]=k

    dt=cl_rslt.copy()
    for k in np.unique(dt):
        dt[dt==k]=np.mean(big_data[dt==k])

    np.save('final_dt.npy',dt)