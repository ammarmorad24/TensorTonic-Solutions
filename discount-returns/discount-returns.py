def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    ans = []
    for i in range(len(rewards)):
        dr = 0
        for index , r in enumerate(rewards[i:]):
            dr += (gamma**(index)) * r
        ans.append(dr)
    return ans


            
    # Write code here