from pydantic import BaseModel,EmailStr,Field
from typing  import Optional


class Student(BaseModel):
    name: str = 'None'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(ge=0,lt=10,default=6.5,description='A decimal value representation')


new_student = {'name':'Subham','age':25,'email':'abc@example.com'}
student = Student(**new_student)
print(student)
print(type(student))
print(student.model_dump_json())