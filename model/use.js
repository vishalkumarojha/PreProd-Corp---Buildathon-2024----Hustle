'use client';

import { useState, useEffect } from "react";

export const useAuth = () => {
    const [user, setUser] = useState({});
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchUser = async () => {
            try {
                const response = await fetch("/api/protected/auth");
                if (response.ok) {
                    const data = await response.json();
                    setUser(data);
                }
            } catch (error) {
                console.error("Error fetching user", error);
            } finally {
                setLoading(false);
            }
        };

        fetchUser();
    }, []);

    return [user, loading];
};