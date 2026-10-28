"""Plateau policy: five consecutive epochs without a new global validation best."""
def advance(state,score,batch_count,patience=5):
    assert 0<=score<=1 and 0<=state['batch_index']<batch_count and patience>0
    if score>state['best']+1e-12:
        state.update(best=score,stale=0)
    else:
        state['stale']+=1
    if state['stale']>=patience:
        state['batch_index']+=1
        state['stale']=0
    return state['batch_index']>=batch_count
