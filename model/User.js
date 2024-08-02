import { Schema, models, model } from "mongoose";

const userSchema = new Schema({
    email: {
        type: String,
        required: true,
        unique: true,
    },
    password: {
        type: String,
        required: true,
    },
    created: {
        type: String,
        default: new Date().toISOString(),
    },
    lastLogin: {
        type: String,
    },
});
const User = models.User || model("User", userSchema, 'users');
export default User;