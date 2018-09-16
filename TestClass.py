class Person:
    _age = 24

    def __init__(self, first_name):
        self.first_name = first_name

# python decorator
class Vehicle:
    def __init__(self, number_of_wheels, type_of_tank, seating_capacity, maximum_velocity):
        self._number_of_wheels = number_of_wheels
        self._type_of_tank = type_of_tank
        self._seating_capacity = seating_capacity
        self._maximum_velocity = maximum_velocity

    @property
    def number_of_wheels(self):
        return self._number_of_wheels

    @number_of_wheels.setter
    def number_of_wheels(self, number):
        self._number_of_wheels = number


tesla_model_s = Vehicle(4, 'electric', 5, 250)
print(tesla_model_s.number_of_wheels) # 4
tesla_model_s.number_of_wheels = 2 # setting number of wheels to 2
print(tesla_model_s.number_of_wheels) # 2