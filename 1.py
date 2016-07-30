class Parent(object):
    def __init__(self):
        self.age = 18
    def implicit(self):
        print 'In class Parent, in func. [implicit]'

class Child(Parent):
    def die(self):
        print 'died'

    def implicit(self, para):
        print 'first we call parent\'s func.'
        super(Child, self).implicit()
        print 'then we use Child\'s func.'
        print ('\n param is : %f' % para)


tom = Parent()
peter = Child()

print tom.age
tom.implicit()
peter.implicit(1.345)
print peter.age
