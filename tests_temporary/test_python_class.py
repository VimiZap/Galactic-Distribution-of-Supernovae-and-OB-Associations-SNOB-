class Student:
    school = 'ABC School'  # class variable, initialized once at the time of import, shared by all instances unless modified (and then it should really be an instance variable)
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def print_student(self):
        print(f'{self.name} is {self.age} years old and attends {self.school}')

John = Student('John', 17)
Mari = Student('Mari', 16)
John.print_student()
Mari.print_student()
John.school = "Springfield Elementary"
John.print_student()
Mari.print_student()