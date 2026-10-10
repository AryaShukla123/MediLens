from pydantic import BaseModel, EmailStr, field_validator, model_validator


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: str | None = None

    class Config:
        from_attributes = True  # allows creating this from a SQLAlchemy User object


class PasswordChange(BaseModel):
    """Used by the profile page's "Change password" form."""
    current_password: str
    new_password: str
    confirm_password: str

    @field_validator("new_password")
    @classmethod
    def new_password_rules(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("New password must be at least 8 characters long")
        if len(v) > 128:
            raise ValueError("New password must be 128 characters or fewer")
        return v

    @model_validator(mode="after")
    def check_passwords(self):
        if self.new_password != self.confirm_password:
            raise ValueError("The new passwords don't match")
        if self.new_password == self.current_password:
            raise ValueError("Your new password must be different from the current one")
        return self
