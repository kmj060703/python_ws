# src/analysis/features/scaler.py
import numpy as np
import pandas as pd

class MinMaxScaler100:
	def __init__(self, clip=True):
		self.min_ = {}
		self.max_ = {}
		self.clip = clip
	
	def fit(self, df, columns):
		for col in columns:
			self.min_[col] = df[col].min()
			self.max_[col] = df[col].max()
		return self
	
	def transform(self, df, columns):
		df_scaled = df.copy()
		for col in columns:
			min_val, max_val = self.min_[col], self.max_[col]
			if max_val == min_val:
				df_scaled[col] = 0.0
			else:
				df_scaled[col] = 100 * (df[col] - min_val) / (max_val - min_val)

			# ✅ 핵심: 0~100 범위 유지 (정책 +%로 max 넘어가도 튀지 않게)
			if self.clip:
				df_scaled[col] = df_scaled[col].clip(0, 100)

		return df_scaled
	
	def fit_transform(self, df, columns):
		return self.fit(df, columns).transform(df, columns)
