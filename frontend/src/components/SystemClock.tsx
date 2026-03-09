import { useState, useEffect } from 'react';
import { getTime } from '../lib/api';

export function SystemClock() {
    const [time, setTime] = useState<string>('');
    const [mode, setMode] = useState<string>('');

    useEffect(() => {
        const fetchTime = async () => {
            try {
                const data = await getTime();
                // Assuming backend returns a string or timestamp we can format, 
                // but requirements say "YYYY-MM-DD HH:mm:ss" and backend returns `current_time`.
                // If backend returns formatted string, we use it. If timestamp, we format.
                // Let's assume backend returns ISO string or similar.
                // User requirement: "get_time format YYYY-MM-DD HH:mm:ss"
                // Let's format it client side if needed, or display as is if already formatted.
                // For safety, let's try to parse and format if it looks like a date.
                // But backend `get_current_time` might be simulation time.
                // Let's just display what backend sends for now, or format if it's a standard timestamp.
                // Remove milliseconds if present (split by dot)
                setTime(data.current_time.split('.')[0]);
                setMode(data.mode);
            } catch (e) {
                console.error("Failed to fetch time", e);
            }
        };

        fetchTime();
        const interval = setInterval(fetchTime, 3000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="fixed top-4 left-4 z-50 bg-white/80 backdrop-blur-md px-4 py-2 rounded-lg border border-slate-200 shadow-md font-display text-sm text-cyan-700">
            <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span className="font-medium tracking-wide">{time || 'Loading...'}</span>
                {mode && <span className="text-xs text-slate-400 font-normal border border-slate-200 px-1.5 py-0.5 rounded bg-slate-50 uppercase tracking-wider">{mode}</span>}
            </div>
        </div>
    );
}
