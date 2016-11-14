#Model Helper

import tflearn

def evaluate_model(model,testX,testY):
	# Evaluate accuracy.
	accuracy_score = model.evaluate(testX,testY,batch_size=32)
	print('Accuracy: %s' %accuracy_score)
	return accuracy_score

def train_model(model, X, Y, epochs, runId, save_path_file):
	#runs more epochs of training on the model
	print ('print model')
	model.fit(X, Y, 
		n_epoch=epochs, 
		shuffle=True,
		show_metric=True, 
		batch_size=32, 
		snapshot_step=500,
		snapshot_epoch=True, 
		run_id=runId,
		validation_set=0.0)
	# Save the model
	model.save(save_path_file)
	return model
