import React, { createContext, useContext, useState } from 'react';
import type { User } from '../types';

interface UserContextType {
  user: User | null;
  tempRegistrationData: User | null;
  setUser: (user: User | null) => void;
  setTempRegistrationData: (data: User | null) => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export function UserProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [tempRegistrationData, setTempRegistrationData] = useState<User | null>(null);

  return (
    <UserContext.Provider value={{ user, tempRegistrationData, setUser, setTempRegistrationData }}>
      {children}
    </UserContext.Provider>
  );
}

export function useUser() {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
}