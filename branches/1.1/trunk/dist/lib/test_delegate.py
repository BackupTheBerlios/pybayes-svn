# this files checks how much time is spent in delegating to lower classes
# compared to the time needed to perform the real operation

# results show that delegating through 4 classes can make the program up to
# 20 times slower...
# plus, it becomes difficult to rewrite the code into C++ and creates a lot of
# programming errors because we don't who delegated to who...

# we should try to avoid this kind of techniques

import numarray as na
from timeit import Timer

class A:
    cpt = na.arange(10)

    def __getitem__(self, index):
        return self.cpt[index]
        # return self.cpt.__getitem__(index) is slower...

    def __setitem__(self, index, value):
        self.cpt[index] = value
        #self.cpt.__setitem__(index,value) is slower...
        
class B:
    a = A()

    def __getitem__(self,index):
        # there is no significative difference between a[index]
        # and a.__getitem__(index) in terms of speed
        # __getitem__ and __setitem__ are just a little bit faster
        
        #return self.a[index]
        return self.a.__getitem__(index)

    def __setitem__(self, index, value):
        #self.a[index] = value
        self.a.__setitem__(index,value)

class C(B):
    a = B()
class D(B):
    a = C()


if __name__ == '__main__':
    d=D()

    print 'Test __getitem__'
    print 'with 4 delegations:',
    print Timer('d[5]','from test_delegate import A,B,C,D;d = D()').timeit(100000)
    print 'with 1 delegations:',
    print Timer('d[5]','from test_delegate import A,B,C,D;d = A()').timeit(100000)
    print 'with no delegation:',
    print Timer('d[5]','from test_delegate import A,B,C,D,na;d = na.arange(10)').timeit(100000)
    
    print 'Test __setitem__'
    print 'with 4 delegations:',
    print Timer('d[5]=1','from test_delegate import A,B,C,D;d = D()').timeit(100000)
    print 'with 1 delegations:',
    print Timer('d[5]=1','from test_delegate import A,B,C,D;d = A()').timeit(100000)
    print 'with no delegation:',
    print Timer('d[5]=1','from test_delegate import A,B,C,D,na;d = na.arange(10)').timeit(100000)

