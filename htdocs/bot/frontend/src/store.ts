import { configureStore } from '@reduxjs/toolkit';

const store = configureStore({
  reducer: {
    // Add your reducers here
  },
});

export default store;
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
