import React from 'react';
import { Link } from 'react-router-dom';
import { ChefHat, Heart, Home, Info, LogOut, User } from 'lucide-react';

interface NavbarProps {
  username?: string;
  onLogout: () => void;
}

export function Navbar({ username, onLogout }: NavbarProps) {
  return (
    <nav className="bg-emerald-600 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <ChefHat className="w-8 h-8" />
            <span className="text-xl font-bold">Food Muse</span>
          </div>

          <div className="flex items-center space-x-8">
            <NavLink to="/" icon={<Home className="w-5 h-5" />} text="Home" />
            <NavLink to="/about" icon={<Info className="w-5 h-5" />} text="About" />
            <NavLink
              to="/saved-recipes"
              icon={<Heart className="w-5 h-5" />}
              text="Saved Recipes"
            />

            {/* <div className="relative group">
              <button className="flex items-center space-x-2 hover:text-emerald-200">
                <User className="w-5 h-5" />
                <span>{username || 'Profile'}</span>
              </button> */}

              {/* <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 hidden group-hover:block">
                <button
                  onClick={onLogout}
                  className="flex items-center px-4 py-2 text-gray-700 hover:bg-emerald-50 w-full"
                >
                  <LogOut className="w-4 h-4 mr-2" />
                  Logout
                </button>
              </div> */}
            {/* </div> */}
          </div>
        </div>
      </div>
    </nav>
  );
}

interface NavLinkProps {
  to: string;
  icon: React.ReactNode;
  text: string;
}

function NavLink({ to, icon, text }: NavLinkProps) {
  return (
    <Link
      to={to}
      className="flex items-center space-x-1 hover:text-emerald-200"
    >
      {icon}
      <span>{text}</span>
    </Link>
  );
}

export default Navbar;