import mongoose from "mongoose";

const DEFAULT_MONGO_URI = "mongodb://localhost:27017/automl";

export default async function connectToDB() {
    if (mongoose.connections[0].readyState) return;

    await mongoose.connect(process.env.MONGO_URI || DEFAULT_MONGO_URI, {
        useNewUrlParser: true,
        useUnifiedTopology: true,
    }).then(() => {
        console.log("Connected to MongoDB");
    }).catch((error) => {
        console.error("Error connecting to MongoDB", error);
    });
}