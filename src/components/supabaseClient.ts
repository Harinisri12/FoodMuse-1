import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = 'https://nhyxleqxhraavroasqpg.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5oeXhsZXF4aHJhYXZyb2FzcXBnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDExNzk1NTUsImV4cCI6MjA1Njc1NTU1NX0._c4l9Wjxl5vVp0LqXadaa9olP8y-ecfADnK18uK9GnU';

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
