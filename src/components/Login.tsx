import React, { useState } from 'react';
import { Lock, Mail } from 'lucide-react';
import { validateEmail } from '../utils/validation';
import type { ValidationError } from '../types';
import { supabase } from './supabaseClient';



interface LoginProps {
  onLogin: (email: string, password: string) => void;
  onSwitchToRegister: () => void;
}

export function Login({ onLogin, onSwitchToRegister }: LoginProps) {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [errors, setErrors] = useState<ValidationError[]>([]);

  const validateForm = (): boolean => {
    const newErrors: ValidationError[] = [];
    
    const emailError = validateEmail(formData.email);
    if (emailError) newErrors.push({ field: 'email', message: emailError });
    if (!formData.password) newErrors.push({ field: 'password', message: 'Password is required' });

    setErrors(newErrors);
    return newErrors.length === 0;
  };


const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();
  
  if (!validateForm()) return;

  const { email, password } = formData;

  // Authenticate with Supabase
  const { error } = await supabase.auth.signInWithPassword({ email, password });

  if (error) {
    setErrors([{ field: 'password', message: 'Invalid email or password' }]);
    return;
  }

  console.log('Login successful');
  onLogin(email, password);
};


  const getError = (field: string) => errors.find(error => error.field === field)?.message;

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <h2 className="mt-6 text-3xl font-extrabold text-gray-900">Sign in to Food Muse</h2>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="rounded-md shadow-sm space-y-4">
            <div>
              <label htmlFor="email" className="sr-only">Email address</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Mail className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="email"
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className="appearance-none rounded-lg relative block w-full pl-10 px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-emerald-500 focus:border-emerald-500"
                  placeholder="Email address"
                />
              </div>
              {getError('email') && (
                <p className="mt-1 text-sm text-red-600">{getError('email')}</p>
              )}
            </div>

            <div>
              <label htmlFor="password" className="sr-only">Password</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-gray-400" />
                </div>
                <input
                  id="password"
                  type="password"
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  className="appearance-none rounded-lg relative block w-full pl-10 px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-emerald-500 focus:border-emerald-500"
                  placeholder="Password"
                />
              </div>
              {getError('password') && (
                <p className="mt-1 text-sm text-red-600">{getError('password')}</p>
              )}
            </div>
          </div>

          <div>
            <button
              type="submit"
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-emerald-600 hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500"
            >
              Sign in
            </button>
          </div>

          <div className="text-center">
            <button
              type="button"
              onClick={onSwitchToRegister}
              className="text-sm text-emerald-600 hover:text-emerald-500"
            >
              Don't have an account? Sign up
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}