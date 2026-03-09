import { PostCard } from './PostCard';
import type { Post } from '../types';

import { Loader2 } from 'lucide-react';

interface PostListProps {
  posts: Post[];
  systemTime?: number;
  onLoadMore?: () => void;
  hasMore?: boolean;
  isLoadingMore?: boolean;
}

export function PostList({ posts, systemTime, onLoadMore, hasMore, isLoadingMore }: PostListProps) {
  // TODO: Implement react-window virtualization for > 100 posts
  // Currently using standard rendering for simplicity and correctness of variable height content.
  
  return (
    <div className="space-y-4">
      {posts.map((post) => (
        <PostCard key={post.id} post={post} systemTime={systemTime} />
      ))}
      
      {posts.length > 0 && hasMore && (
        <div className="text-center pt-4 pb-8">
          <button 
            onClick={onLoadMore}
            disabled={isLoadingMore}
            className="px-6 py-2 bg-white border border-slate-200 rounded-full text-slate-600 font-medium text-sm hover:bg-slate-50 hover:border-cyan-200 hover:text-cyan-600 transition-all shadow-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 mx-auto"
          >
            {isLoadingMore && <Loader2 size={14} className="animate-spin" />}
            {isLoadingMore ? 'Loading...' : 'Load More'}
          </button>
        </div>
      )}

      {posts.length === 0 && (
        <div className="text-center py-20 bg-white/50 backdrop-blur rounded-xl border border-slate-100 shadow-sm">
          <div className="text-4xl mb-3">📭</div>
          <h3 className="text-lg font-bold text-slate-700 mb-1">No posts yet</h3>
          <p className="text-slate-500 text-sm">Waiting for new updates...</p>
        </div>
      )}
    </div>
  );
}
